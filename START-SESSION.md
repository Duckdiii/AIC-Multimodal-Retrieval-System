# START SESSION - AIC Competition System v4.0

## **🚀 QUICK START GUIDE**

### **System Status Check**
```bash
# Verify current system status
cd <project-root>
python system.py --status
python gui.py --check-dependencies
```

### **Competition Mode Activation**
```bash
# Switch to AIC Competition Mode
python system.py --mode=competition --version=4.0
```

## **📋 PRE-DEVELOPMENT CHECKLIST**

### **✅ Environment Setup**
- [ ] **Python Environment**: 3.9+ activated with all dependencies
- [ ] **API Keys**: OpenAI API key configured in config.json
- [ ] **Vietnamese Models**: VietOCR and spaCy Vietnamese models installed
- [ ] **System Resources**: Minimum 16GB RAM, GPU with 8GB+ VRAM
- [ ] **Backup**: Current v3.0 system backed up to `legacy/` folder

### **✅ Data Validation**
- [ ] **Sample Queries**: All 24 sample queries (.txt files) accessible in `/sample`
- [ ] **Keyframe Database**: Current keyframe index loaded and functional
- [ ] **Video Collection**: Target video dataset available and indexed
- [ ] **CSV Templates**: Competition CSV format templates prepared

### **✅ Baseline Testing**
```bash
# Test current system with sample queries
python test_baseline_performance.py --sample-queries=sample/
```
Expected results: Current system baseline performance documented

## **🎯 DEVELOPMENT SESSION WORKFLOW**

### **Phase 1: Core Engine Enhancement (Days 1-7)**

#### **Session 1-2: Multi-Scene Query Parser**
**Objective**: Break complex queries into scene components

**Setup Commands**:
```bash
mkdir -p core/
touch core/enhanced_query_parser.py
```

**Development Focus**:
```python
# Sample implementation target
query = "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng..."
parsed = parse_multi_scene_query(query)
# Expected output: structured scene breakdown with temporal markers
```

**Validation Test**:
```bash
python test_query_parser.py --query=sample/query-p1-12-kis.txt
```

#### **Session 3-4: Video-Level Diversification**
**Objective**: Group results by video, prevent single-video dominance

**Current Problem Simulation**:
```bash
# Current system: Returns many frames from same video
python system.py --search="bánh rán trang trí"
# Output: [L01_V028:frame1, L01_V028:frame2, L01_V028:frame3, ...]
# Problem: All results from one video, missing other videos
```

**Target Improvement**:
```python
# New system should return:
# [VideoResult(L01_V028, [frame1,frame2]), VideoResult(L05_V012, [frame3]), ...]
```

#### **Session 5-7: Integration & Testing**
**Objective**: Integrate new components with existing system

**Integration Test**:
```bash
python system_v4_integration.py --test-mode=true --sample-queries=sample/
```

### **Phase 2: Temporal Reasoning (Days 8-14)**

#### **Session 8-10: TRAKE Temporal Engine**
**Objective**: Detect sequential events in TRAKE queries

**Sample TRAKE Query Testing**:
```bash
# Test with sample TRAKE query
python temporal_event_detector.py --query=sample/query-p1-4-trake.txt
# Expected: 4 precise frame numbers representing event sequence
```

**Event Detection Validation**:
```python
# Target: Detect cooking process events
events = [
    "Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây",
    "Khoảnh khắc đầu tiên thấy miến măng tây tiếp xúc với dầu", 
    "Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo",
    "Khoảng khắc miếng măng tây cuối cùng rời chảo"
]
frames = detect_temporal_events(video_path, events)
# Expected: [1200, 1850, 2100, 2450] (chronological order)
```

#### **Session 11-12: Vietnamese Context Enhancement**
**Objective**: Handle Vietnamese OCR and cultural context

**OCR Testing**:
```bash
# Test Vietnamese text recognition
python vietnamese_ocr.py --test-frame=sample_frames/text_overlay.jpg
# Expected: Accurate Vietnamese text extraction
```

**Cultural Context Testing**:
```python
# Test location recognition
question = "Hỏi xã này có tên là gì?"
answer = process_vietnamese_qa(frame_path, question)
# Expected: "Diên Khánh" (or appropriate location name)
```

### **Phase 3: Competition Integration (Days 15-21)**

#### **Session 15-17: Competition Format Engine**
**Objective**: Generate competition-compliant CSV files

**CSV Generation Testing**:
```bash
# Test CSV generation with sample queries
python competition_csv_generator.py --input=sample/ --output=test_results/
```

**Expected Output Structure**:
```
test_results/
├── query-p1-1-kis.csv    (L01_V028,25300)
├── query-p1-15-qa.csv    (L01_V028,3450,"Diên Khánh")  
├── query-p1-4-trake.csv  (L10_V001,1200,1850,2100,2450)
└── ...
```

#### **Session 18-19: End-to-End Testing**
**Objective**: Comprehensive system validation

**Full Pipeline Test**:
```bash
# Complete competition simulation
python test_competition_pipeline.py --mode=full-simulation
```

**Performance Validation**:
```bash
# Benchmark performance
python performance_benchmarks.py --target-metrics=PRODUCTION-REQUIREMENTS.md
```

#### **Session 20-21: Final Optimization**
**Objective**: System optimization and competition readiness

**Final Validation Checklist**:
```bash
# Run comprehensive test suite
python validate_competition_readiness.py --strict-mode=true
```

## **🔬 TESTING & VALIDATION PROTOCOLS**

### **Unit Testing Commands**
```bash
# Test individual components
python -m pytest tests/test_query_parser.py -v
python -m pytest tests/test_video_diversification.py -v
python -m pytest tests/test_temporal_detection.py -v
python -m pytest tests/test_vietnamese_ocr.py -v
python -m pytest tests/test_csv_generation.py -v
```

### **Integration Testing Commands**
```bash
# Test component integration
python test_integration.py --component=query_parser+video_search
python test_integration.py --component=temporal_engine+csv_generator
python test_integration.py --component=full_pipeline
```

### **Performance Testing Commands**
```bash
# Speed benchmarks
python benchmark_speed.py --queries=sample/ --iterations=10

# Memory usage monitoring
python benchmark_memory.py --batch-size=25

# Accuracy validation
python benchmark_accuracy.py --ground-truth=validation_set/
```

## **📊 MONITORING & DEBUGGING**

### **Real-time Monitoring**
```bash
# Monitor system performance during development
python monitor_system.py --log-level=DEBUG --output=logs/development.log
```

### **Debug Mode Activation**
```python
# Enable detailed debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Component-specific debugging
ENABLE_QUERY_PARSER_DEBUG = True
ENABLE_TEMPORAL_ENGINE_DEBUG = True
ENABLE_VIETNAMESE_OCR_DEBUG = True
```

### **Common Debugging Scenarios**

#### **Query Parser Issues**
```bash
# Debug query parsing failures
python debug_query_parser.py --query="problematic query text" --verbose
```

#### **Video Search Issues**
```bash
# Debug video diversification problems
python debug_video_search.py --query="search term" --show-clustering
```

#### **Temporal Detection Issues**
```bash
# Debug TRAKE event detection
python debug_temporal_events.py --video=sample_video.mp4 --events="event list"
```

#### **CSV Format Issues**
```bash
# Validate CSV format compliance
python validate_csv_format.py --file=output.csv --strict-mode
```

## **⚡ RAPID DEVELOPMENT SHORTCUTS**

### **Quick Component Templates**
```bash
# Generate component boilerplate
python generate_component.py --name=new_feature --type=processor
python generate_component.py --name=new_test --type=test
```

### **Fast Testing Cycles**
```bash
# Quick validation with single query
python quick_test.py --query=sample/query-p1-1-kis.txt --component=all

# Rapid CSV validation
python quick_csv_check.py --file=test_output.csv
```

### **Development Utilities**
```bash
# Auto-format code
python format_code.py --target=core/,temporal/,vietnamese/,competition/

# Check dependencies
python check_dependencies.py --requirements=requirements_v4.txt

# Update documentation
python update_docs.py --auto-generate
```

## **🚨 TROUBLESHOOTING GUIDE**

### **Common Issues & Solutions**

#### **Issue: OpenAI API Failures**
```bash
# Check API connectivity
python test_openai_connection.py
# Solution: Verify API key in config.json, check rate limits
```

#### **Issue: Vietnamese OCR Poor Accuracy**
```bash
# Test OCR models
python test_vietnamese_ocr_models.py --compare-engines
# Solution: Try VietOCR vs PaddleOCR, adjust image preprocessing
```

#### **Issue: Memory Errors During Batch Processing**
```bash
# Monitor memory usage
python monitor_memory.py --batch-processing
# Solution: Reduce batch size, implement memory cleanup
```

#### **Issue: Slow Query Processing**
```bash
# Profile performance bottlenecks
python profile_performance.py --query=slow_query.txt
# Solution: Optimize indexing, cache intermediate results
```

#### **Issue: CSV Format Validation Failures**
```bash
# Debug CSV format
python debug_csv_format.py --file=problematic.csv --show-details
# Solution: Check UTF-8 encoding, comma delimiters, quote escaping
```

## **🎯 SUCCESS VALIDATION**

### **Milestone Checkpoints**

#### **Week 1 Success Criteria**
- [ ] Multi-scene query parser working with complex queries
- [ ] Video-diversified search returning balanced results  
- [ ] Vietnamese temporal marker detection functional
- [ ] System integration complete without breaking existing features

#### **Week 2 Success Criteria**
- [ ] TRAKE engine detecting event sequences accurately
- [ ] Vietnamese OCR extracting text from video frames
- [ ] Q&A system answering location/cultural questions
- [ ] Temporal reasoning working for cooking/manufacturing processes

#### **Week 3 Success Criteria**
- [ ] Competition CSV files generated in correct format
- [ ] Batch processing handling 25+ queries efficiently
- [ ] End-to-end pipeline: query files → submission.zip
- [ ] All performance targets met (speed, accuracy, reliability)

### **Final Competition Readiness Check**
```bash
# Ultimate validation test
python final_competition_check.py --comprehensive
```

**Expected Output**:
```
✅ All 24 sample queries processed successfully
✅ CSV format validation: 100% compliant
✅ Performance metrics: All targets exceeded
✅ Submission package: Ready for upload
✅ System reliability: Stable under load
✅ Competition readiness: APPROVED
```

## **📈 SUCCESS METRICS DASHBOARD**

### **Real-time KPI Monitoring**
```bash
# Launch development dashboard
python development_dashboard.py --port=8080
# Access: http://localhost:8080
```

**Dashboard Metrics**:
- Query processing speed (current vs target)
- Accuracy rates by query type (KIS/Q&A/TRAKE)
- System resource utilization
- Test success rates
- Development progress tracking

### **Final Performance Report**
At project completion, generate comprehensive performance report:
```bash
python generate_final_report.py --output=FINAL_PERFORMANCE_REPORT.md
```

This START-SESSION guide provides comprehensive guidance for systematic development of the AIC Competition System v4.0, ensuring efficient progress tracking and successful competition readiness.
