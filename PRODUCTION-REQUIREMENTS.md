# AIC Competition System v4.0 - Production Requirements

## **EXECUTIVE SUMMARY**
Enhanced Retrieval System v4.0 for AIC Video Retrieval Competition - addressing multi-scene queries, temporal reasoning, and video-level result diversification through revolutionary architecture changes.

## **PROBLEM ANALYSIS**

### **Current System Limitations**
```
Query: "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ 
màu trắng nằm trên một khay gỗ hình chữ nhật. Bên cạnh chiếc đĩa sứ là một chén đựng 
một vài trái dâu, nhưng có 2 trái bị rơi ra ngoài..."

Current System → Returns single frame
Reality → Need sequence of frames from same video showing entire process
```

### **Competition Requirements Analysis**

**KIS (Known Item Search)**:
- **Format**: `<video_name>, <frame_id>`  
- **Challenge**: Multi-scene descriptions → need video-level grouping
- **Current Gap**: Single-frame focus, no scene sequence understanding

**Q&A (Question Answering)**:
- **Format**: `<video_name>, <frame_id>, "<answer>"`
- **Challenge**: OCR + Vietnamese cultural context
- **Current Gap**: No text extraction, limited Vietnamese support

**TRAKE (Temporal Retrieval)**:
- **Format**: `<video_name>, <frame1>, <frame2>, <frame3>, <frame4>`
- **Challenge**: Sequential event detection with temporal boundaries
- **Current Gap**: No temporal reasoning capabilities

## **SOLUTION ARCHITECTURE v4.0**

### **1. Multi-Modal Query Parser**
```python
class MultiModalQueryParser:
    def parse_complex_query(self, query: str) -> QueryStructure:
        """
        Break complex queries into structured components
        
        Input: "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu..."
        Output: {
            "main_topic": "trang trí bánh rán",
            "scenes": [
                {"id": 1, "description": "đĩa sứ màu trắng trên khay gỗ"},
                {"id": 2, "description": "chén đựng dâu, 2 trái rơi ra ngoài"},
                {"id": 3, "description": "đặt 2 bánh rán lên đĩa"},
                {"id": 4, "description": "rưới chocolate lên bánh"}
            ],
            "temporal_markers": ["bắt đầu", "tiếp theo", "sau đó"],
            "query_type": "multi_scene_sequence"
        }
        """
```

### **2. Video-Level Diversified Retriever**
```python
class VideoLevelRetriever:
    def search_with_diversification(self, query_structure: QueryStructure) -> List[VideoResult]:
        """
        Revolutionary approach: Group results by video, ensure diversity
        
        Instead of: [Frame1_VideoA, Frame2_VideoA, Frame3_VideoA, ...]
        Return: [
            VideoResult(video="L01_V028", frames=[1234, 1456, 1789], confidence=0.95),
            VideoResult(video="L02_V015", frames=[5643, 5890], confidence=0.87),
            VideoResult(video="L05_V032", frames=[12000], confidence=0.82)
        ]
        """
```

### **3. Temporal Event Detector (TRAKE Engine)**
```python
class TemporalEventDetector:
    def detect_event_sequence(self, video_path: str, events: List[str]) -> List[int]:
        """
        Advanced temporal reasoning for TRAKE queries
        
        Events: [
            "Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây",
            "Khoảnh khắc đầu tiên thấy miến măng tây tiếp xúc với dầu trong chảo",
            "Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu",
            "Khoảng khắc miếng măng tây cuối cùng rời chảo dầu"
        ]
        
        Returns: [frame_1, frame_2, frame_3, frame_4] with temporal ordering
        """
```

### **4. Vietnamese Context Enhancer**
```python
class VietnameseContextEnhancer:
    def __init__(self):
        self.ocr_engine = VietOCR()  # Vietnamese-optimized OCR
        self.location_db = VietnamLocationDB()  # Vietnamese locations database
        self.cultural_context = VietnameseCulturalKnowledge()
        
    def enhance_qa_understanding(self, frame_path: str, question: str) -> str:
        """
        Vietnamese-specific Q&A processing
        
        Example:
        Question: "Hỏi xã này có tên là gì?"
        Frame: [Image with Vietnamese text]
        → OCR → "Xã Diên Khánh" → Answer: "Diên Khánh"
        """
```

## **SYSTEM COMPONENTS v4.0**

### **Core Architecture Enhancement**
```
Input Query → Multi-Scene Parser → Video-Level Search → 
Temporal Analysis → Result Diversification → CSV Generation
```

### **Component Specifications**

#### **1. Enhanced Query Processor v4.0**
- **Multi-Scene Detection**: LLM-based scene decomposition
- **Temporal Annotation**: Identify temporal markers ("bắt đầu", "sau đó", "cuối cùng")
- **Vietnamese NLP**: Optimized for Vietnamese language patterns
- **Complexity Analysis**: Classify query complexity for appropriate processing

#### **2. Video-Centric Search Engine v4.0**
- **Video-Level Grouping**: Cluster results by video to ensure diversity
- **Scene Coverage Analysis**: Ensure multiple scenes within video are covered
- **Confidence Scoring**: Video-level confidence based on scene coverage
- **Diversification Algorithm**: Prevent single video from dominating results

#### **3. Temporal Reasoning Engine v4.0**
- **Event Boundary Detection**: Identify start/end of events in TRAKE
- **Sequential Alignment**: Ensure proper temporal ordering
- **Process Understanding**: Specialized for cooking, manufacturing processes
- **Frame Precision**: Sub-second accuracy for event detection

#### **4. Competition Format Generator v4.0**
- **Automated CSV Export**: Generate competition-compliant CSV files
- **Format Validation**: Ensure UTF-8, comma-delimited, no headers
- **Batch Processing**: Handle entire query sets automatically
- **Submission Packaging**: Auto-generate submission.zip structure

## **IMPLEMENTATION STRATEGY**

### **Phase 1: Core Engine Enhancement (Week 1)**
1. **Multi-Scene Query Parser**
   - Implement LLM-based scene decomposition
   - Add Vietnamese language processing
   - Create temporal marker detection

2. **Video-Level Result Grouping**
   - Modify FAISS search to group by video
   - Implement diversification algorithms
   - Add video-level confidence scoring

### **Phase 2: Temporal Reasoning (Week 2)**
3. **TRAKE Engine Development**
   - Build event boundary detection
   - Implement temporal alignment algorithms
   - Add process-specific understanding (cooking, etc.)

4. **Vietnamese Context Enhancement**
   - Integrate Vietnamese OCR (VietOCR)
   - Build Vietnamese location/cultural database
   - Enhance Q&A processing

### **Phase 3: Competition Integration (Week 3)**
5. **Format Compliance**
   - Build CSV generation engine
   - Implement batch query processing
   - Add submission packaging automation

6. **Testing & Optimization**
   - End-to-end testing with sample queries
   - Performance optimization
   - Accuracy validation

## **TECHNICAL SPECIFICATIONS**

### **Input/Output Formats**

#### **Input Processing**
```python
# Query Files
query_files = {
    "kis": ["query-1-kis.txt", "query-2-kis.txt", ...],
    "qa": ["query-1-qa.txt", "query-2-qa.txt", ...],
    "trake": ["query-1-trake.txt", "query-2-trake.txt", ...]
}

# Batch Processing
results = process_query_batch(query_files)
```

#### **Output Generation**
```python
# KIS Output Format
kis_results = [
    ("L01_V028", 25300),
    ("L02_V015", 14500),
    ("L05_V032", 8900)
]

# Q&A Output Format  
qa_results = [
    ("L01_V028", 3450, "Diên Khánh"),
    ("L02_V011", 1200, "Năm người"),
    ("L03_V005", 2800, "Màu đỏ")
]

# TRAKE Output Format
trake_results = [
    ("L10_V001", 1200, 1850, 2100, 2450),
    ("L11_V003", 5100, 5700, 6200, 6800)
]
```

### **Performance Targets**

#### **Speed Requirements**
- **Query Processing**: <5 seconds per query (simple), <15 seconds (complex)
- **Batch Processing**: 25 queries in <10 minutes
- **CSV Generation**: <1 second per file

#### **Accuracy Targets**
- **KIS**: Top-10 accuracy >85% (multi-scene queries)
- **Q&A**: Combined frame+answer accuracy >75%
- **TRAKE**: Event sequence accuracy >80%

#### **System Reliability**
- **Memory Efficiency**: Process 100+ videos without memory issues
- **Error Recovery**: Graceful handling of failed queries
- **Format Compliance**: 100% CSV specification adherence

## **USER WORKFLOW ENHANCEMENT**

### **Current Workflow (Problematic)**
```
1. User inputs query
2. System returns 100 scattered frames
3. User manually checks each frame
4. High chance of missing correct video (out of display range)
5. Manual CSV creation
```

### **Enhanced Workflow v4.0**
```
1. User inputs query (or batch of queries)
2. System automatically:
   - Parses multi-scene components
   - Returns video-grouped results
   - Generates CSV files
3. User reviews video-level results (much easier verification)
4. System auto-packages submission.zip
5. Ready for competition upload
```

## **SUCCESS METRICS**

### **User Experience Improvements**
- **Verification Time**: 80% reduction in manual checking time
- **Result Relevance**: Video-diversified results prevent missing correct answers
- **Automation Level**: 90% automated pipeline (query → submission)

### **Competition Performance**
- **Submission Accuracy**: Zero format/compliance errors
- **Processing Speed**: Handle competition workload efficiently
- **Result Quality**: Consistent top-tier performance across query types

### **System Capabilities**
- **Multi-Scene Understanding**: Handle complex sequential descriptions
- **Temporal Reasoning**: Accurate event boundary detection
- **Cultural Context**: Vietnamese-optimized OCR and cultural knowledge
- **Scalability**: Process entire competition datasets efficiently

## **RISK MITIGATION**

### **Technical Risks**
- **LLM Dependency**: Implement fallback parsers for LLM failures
- **Vietnamese OCR**: Multiple OCR engines for redundancy
- **Temporal Complexity**: Graduated complexity handling (simple → advanced)

### **Competition Risks**
- **Format Compliance**: Comprehensive validation testing
- **Performance**: Load testing with competition-scale data
- **Accuracy**: Validation against known ground truth samples

## **DEPLOYMENT STRATEGY**

### **Testing Environment**
- **Sample Queries**: Validate with provided sample queries
- **Performance Benchmarking**: Measure against current system
- **Format Validation**: Ensure CSV compliance

### **Production Deployment**
- **Backup Systems**: Keep current system as fallback
- **Monitoring**: Real-time performance tracking
- **Quick Rollback**: Rapid revert capability if issues arise

This production requirements document provides the comprehensive framework for developing AIC Competition System v4.0, addressing all identified limitations and competition requirements through innovative architectural enhancements.