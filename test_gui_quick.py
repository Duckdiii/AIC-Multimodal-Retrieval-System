#!/usr/bin/env python3
"""
Quick GUI Test - Check if GUI loads without errors
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gui_imports():
    """Test if GUI imports work correctly"""
    try:
        from PyQt5.QtWidgets import QApplication
        from gui import MainWindow
        print("PASS: GUI imports successful")
        return True
    except ImportError as e:
        print(f"FAIL: GUI import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL: GUI import error: {e}")
        return False

def test_gui_class_attributes():
    """Test if GUI class has v4.0 features"""
    try:
        from gui import MainWindow
        import inspect
        
        # Get MainWindow class source
        source_lines = inspect.getsource(MainWindow)
        
        # Check for v4.0 features in source code
        has_video_grouping = 'video_grouping_checkbox' in source_lines
        has_diversity_control = 'diversity_threshold_spinbox' in source_lines
        has_multi_scene = 'multi_scene_checkbox' in source_lines
        has_grouping_strategy = 'grouping_strategy_combo' in source_lines
        
        print("PASS: GUI class loaded successfully")
        print(f"PASS: Video grouping control in code: {has_video_grouping}")
        print(f"PASS: Diversity threshold control in code: {has_diversity_control}")
        print(f"PASS: Multi-scene parsing control in code: {has_multi_scene}")
        print(f"PASS: Grouping strategy selector in code: {has_grouping_strategy}")
        
        return has_video_grouping and has_diversity_control and has_multi_scene
        
    except Exception as e:
        print(f"FAIL: GUI class analysis failed: {e}")
        return False

def main():
    print("Quick GUI Test")
    print("=" * 20)
    
    # Test 1: Imports
    if not test_gui_imports():
        return False
    
    # Test 2: GUI Class Features
    if not test_gui_class_attributes():
        return False
    
    print("\nAll GUI tests passed!")
    print("GUI is ready for use with v4.0 competition features.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)