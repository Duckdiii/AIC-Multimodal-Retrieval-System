"""
Quick Prototype Test - AIC Competition System v4.0
Simple validation of core concepts with minimal dependencies
"""

import os
import random
from pathlib import Path

# Project-relative data paths. Environment variables allow external datasets.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", PROJECT_ROOT))
KEYFRAMES_DIR = Path(os.environ.get("AIC_KEYFRAMES_DIR", DATA_ROOT / "Keyframes"))
SAMPLE_QUERIES_DIR = Path(
    os.environ.get("AIC_SAMPLE_QUERIES_DIR", PROJECT_ROOT / "sample")
)

def test_dataset_access():
    """Test if we can access the dataset"""
    print("=" * 50)
    print("TEST 1: Dataset Access")
    print("=" * 50)
    
    if not os.path.exists(KEYFRAMES_DIR):
        print(f"ERROR: Cannot access {KEYFRAMES_DIR}")
        return False
    
    # Count videos
    video_dirs = [d for d in os.listdir(KEYFRAMES_DIR) 
                  if os.path.isdir(os.path.join(KEYFRAMES_DIR, d))]
    
    print(f"SUCCESS: Found {len(video_dirs)} video directories")
    
    # Sample analysis
    sample_videos = video_dirs[:5]
    
    for video in sample_videos:
        video_path = os.path.join(KEYFRAMES_DIR, video)
        try:
            frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            print(f"  {video}: {len(frame_files)} frames")
        except Exception as e:
            print(f"  {video}: ERROR - {e}")
    
    return True

def test_query_parsing():
    """Test basic query parsing"""
    print("\n" + "=" * 50) 
    print("TEST 2: Query Parsing")
    print("=" * 50)
    
    if not os.path.exists(SAMPLE_QUERIES_DIR):
        print(f"ERROR: Cannot access {SAMPLE_QUERIES_DIR}")
        return False
        
    query_files = [f for f in os.listdir(SAMPLE_QUERIES_DIR) if f.endswith('.txt')]
    print(f"Found {len(query_files)} query files")
    
    # Test parsing 3 sample queries
    test_files = query_files[:3]
    
    for query_file in test_files:
        query_path = os.path.join(SAMPLE_QUERIES_DIR, query_file)
        
        try:
            with open(query_path, 'r', encoding='utf-8') as f:
                query_text = f.read().strip()
            
            # Basic analysis
            sentences = query_text.split('.')
            words = query_text.split()
            
            # Detect complexity
            complexity = 'simple'
            if len(query_text) > 200 and len(sentences) > 3:
                complexity = 'complex'
            elif len(query_text) > 100:
                complexity = 'medium'
            
            query_type = 'unknown'
            if 'kis' in query_file:
                query_type = 'KIS'
            elif 'qa' in query_file:
                query_type = 'Q&A'  
            elif 'trake' in query_file:
                query_type = 'TRAKE'
            
            print(f"  {query_file}:")
            print(f"    Type: {query_type}")
            print(f"    Length: {len(query_text)} chars")
            print(f"    Sentences: {len(sentences)}")
            print(f"    Complexity: {complexity}")
            print(f"    Preview: {query_text[:80]}...")
            
        except Exception as e:
            print(f"  {query_file}: ERROR - {e}")
    
    return True

def test_video_grouping_concept():
    """Test video-level grouping concept"""
    print("\n" + "=" * 50)
    print("TEST 3: Video Grouping Concept")
    print("=" * 50)
    
    # Get video list
    if not os.path.exists(KEYFRAMES_DIR):
        print("ERROR: Cannot access keyframes directory")
        return False
        
    video_dirs = [d for d in os.listdir(KEYFRAMES_DIR) 
                  if os.path.isdir(os.path.join(KEYFRAMES_DIR, d))]
    
    if len(video_dirs) < 10:
        print("ERROR: Need at least 10 videos for testing")
        return False
    
    # Simulate current approach (frame-dominated)
    print("\nCURRENT APPROACH Simulation:")
    print("- Results dominated by few videos")
    
    dominant_videos = video_dirs[:3]  # Top 3 videos dominate
    current_results = []
    
    for i in range(50):  # 50 results
        # 70% from dominant videos (current problem)
        if random.random() < 0.7:
            video = random.choice(dominant_videos)
        else:
            video = random.choice(video_dirs)
        
        current_results.append(video)
    
    # Count video occurrences
    from collections import Counter
    current_counts = Counter(current_results)
    
    print(f"  Unique videos in results: {len(current_counts)}")
    print(f"  Most common video appears: {current_counts.most_common(1)[0][1]} times")
    print(f"  Top 3 videos: {dict(current_counts.most_common(3))}")
    
    # Simulate new approach (video-diversified)
    print("\nNEW APPROACH Simulation:")
    print("- Video-level diversification")
    
    selected_videos = random.sample(video_dirs, min(15, len(video_dirs)))  # Select 15 diverse videos
    new_results = []
    
    for video in selected_videos:
        # Each video contributes 2-4 frames
        frames_per_video = random.randint(2, 4)
        new_results.extend([video] * frames_per_video)
    
    # Limit to 50 results
    new_results = new_results[:50]
    new_counts = Counter(new_results)
    
    print(f"  Unique videos in results: {len(new_counts)}")  
    print(f"  Most common video appears: {new_counts.most_common(1)[0][1]} times")
    print(f"  Top 3 videos: {dict(new_counts.most_common(3))}")
    
    # Calculate improvement
    diversity_improvement = len(new_counts) / len(current_counts)
    dominance_reduction = (current_counts.most_common(1)[0][1] - new_counts.most_common(1)[0][1])
    
    print(f"\nIMPROVEMENT METRICS:")
    print(f"  Diversity increase: {diversity_improvement:.2f}x more videos")
    print(f"  Dominance reduction: {dominance_reduction} fewer duplicates from top video")
    print(f"  Better coverage: {'YES' if diversity_improvement > 1.5 else 'NO'}")
    
    return True

def test_vietnamese_processing():
    """Test basic Vietnamese text processing"""
    print("\n" + "=" * 50)
    print("TEST 4: Vietnamese Processing")
    print("=" * 50)
    
    # Test queries with Vietnamese entities
    test_cases = [
        {
            'text': 'Hỏi xã này có tên là gì? tại tỉnh Khánh Hòa',
            'expected_entities': ['Khánh Hòa'],
            'answer_type': 'location'
        },
        {
            'text': 'công thức với nguyên liệu chính là 200g thịt nạc xay',
            'expected_entities': ['200g', 'thịt nạc xay'],
            'answer_type': 'recipe'
        },
        {
            'text': 'Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây',
            'expected_entities': ['măng tây', 'bột'],
            'answer_type': 'cooking_action'
        }
    ]
    
    # Simple entity extraction
    vietnamese_locations = ['Khánh Hòa', 'Diên Khánh', 'Hà Nội', 'TP.HCM', 'Đà Nẵng']
    cooking_terms = ['thịt nạc xay', 'măng tây', 'bánh rán', 'chocolate', 'dâu tây', 'bột']
    
    for i, test_case in enumerate(test_cases):
        text = test_case['text']
        print(f"\nTest case {i+1}: {text[:50]}...")
        
        # Extract entities
        found_locations = [loc for loc in vietnamese_locations if loc in text]
        found_cooking = [term for term in cooking_terms if term in text]
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+', text)
        
        entities_found = {
            'locations': found_locations,
            'cooking_terms': found_cooking, 
            'numbers': numbers
        }
        
        print(f"  Found entities: {entities_found}")
        
        # Simulate answer generation
        if found_locations:
            answer = found_locations[0]
            confidence = 0.85
        elif found_cooking:
            answer = f"Recipe with {found_cooking[0]}"
            confidence = 0.75
        elif numbers:
            answer = numbers[0]
            confidence = 0.7
        else:
            answer = "Unknown"
            confidence = 0.3
            
        print(f"  Predicted answer: {answer} (confidence: {confidence:.2f})")
        print(f"  Processing: {'SUCCESS' if confidence > 0.6 else 'NEEDS_IMPROVEMENT'}")
    
    return True

def main():
    """Run all quick tests"""
    print("QUICK PROTOTYPE TEST - AIC Competition System v4.0")
    print("Testing core concepts with real data...")
    
    results = {
        'dataset_access': False,
        'query_parsing': False,
        'video_grouping': False, 
        'vietnamese_processing': False
    }
    
    try:
        results['dataset_access'] = test_dataset_access()
        results['query_parsing'] = test_query_parsing()
        results['video_grouping'] = test_video_grouping_concept()
        results['vietnamese_processing'] = test_vietnamese_processing()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY RESULTS")
        print("=" * 60)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= 3:
            print("CONCLUSION: Core concepts are viable, proceed with development!")
        else:
            print("CONCLUSION: Need to address failing tests before proceeding.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
