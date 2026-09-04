#!/usr/bin/env python3
"""
Quick Index Builder for v4.0 System
Build FAISS index and metadata to enable search functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_system_index():
    """Build index for the retrieval system"""
    print("Building System Index for v4.0")
    print("=" * 32)
    
    try:
        # Import system
        from system import EnhancedRetrievalSystem
        
        print("1. Initializing Enhanced Retrieval System...")
        system = EnhancedRetrievalSystem(auto_initialize=True, verbose=True)
        
        # Check keyframes directory
        keyframes_path = Path("keyframes")
        if not keyframes_path.exists():
            print("ERROR: keyframes directory not found")
            return False
            
        video_count = len([d for d in keyframes_path.iterdir() if d.is_dir()])
        print(f"2. Found {video_count} video directories in keyframes/")
        
        if video_count == 0:
            print("ERROR: No video directories found in keyframes/")
            return False
        
        # Build index
        print("3. Building FAISS index and metadata...")
        print("   This may take several minutes...")
        
        # Use build_system method with keyframe folder
        success = system.build_system("keyframes")
        
        if success:
            print("4. Index build completed successfully!")
            
            # Verify index
            print("5. Verifying index...")
            health = system.get_system_health()
            
            if health.get("is_healthy", False):
                print("   SUCCESS: System is healthy and ready for search")
                
                # Print summary
                index_info = health.get("components", {}).get("faiss", {})
                metadata_info = health.get("components", {}).get("metadata", {})
                
                print("\n   Index Summary:")
                print(f"   - FAISS index: {index_info.get('status', 'unknown')}")
                print(f"   - Metadata: {metadata_info.get('status', 'unknown')}")
                
                return True
            else:
                issues = health.get("issues", [])
                print(f"   WARNING: System health check found issues: {issues}")
                return False
        else:
            print("4. Index build failed!")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Quick Index Builder for Enhanced Retrieval System v4.0")
    print("=" * 55)
    
    # Check if keyframes exist
    if not Path("keyframes").exists():
        print("ERROR: keyframes directory not found!")
        print("Please ensure you have video keyframes in the keyframes/ directory")
        return False
    
    # Build index
    success = build_system_index()
    
    if success:
        print("\nSUCCESS: Index build completed!")
        print("The system is now ready for v4.0 search functionality.")
        print("\nNext steps:")
        print("1. Test search through GUI")
        print("2. Try v4.0 features: video grouping, multi-scene parsing")
        return True
    else:
        print("\nFAILED: Index build unsuccessful")
        print("Please check the error messages above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)