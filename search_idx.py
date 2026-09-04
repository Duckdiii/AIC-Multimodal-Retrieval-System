"""
Search Index Tool for Video Frame Lookup
========================================

This tool allows users to:
1. Select a folder containing videos
2. Search for a specific video by name  
3. Input a time in seconds
4. Get the corresponding frame_index at that timestamp

Features:
- Browse folder dialog for video selection
- Video validation and metadata extraction
- Time-based frame index calculation
- Support for common video formats (mp4, avi, mkv, mov, etc.)
- Error handling and user-friendly interface

Author: Video Frame Search Tool
Version: 1.0
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import threading


class VideoFrameSearchTool:
    """Main class for video frame search functionality"""
    
    def __init__(self):
        self.video_folder = ""
        self.video_info = {}  # Cache for video information
        self.supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Video Frame Search Tool - ONE_FOR_ALL_v3.0")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎥 Video Frame Search Tool", 
                              font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(main_frame, text="1. Select Video Folder", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Button(folder_frame, text="Browse Folder", 
                  command=self.browse_folder).grid(row=0, column=0, padx=(0, 10))
        
        self.folder_label = ttk.Label(folder_frame, text="No folder selected", 
                                     foreground='gray')
        self.folder_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Video list section  
        list_frame = ttk.LabelFrame(main_frame, text="2. Available Videos", padding="10")
        list_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        
        # Video listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        
        self.video_listbox = tk.Listbox(list_container, height=8, font=('Consolas', 10))
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        
        # Search section
        search_frame = ttk.LabelFrame(main_frame, text="3. Search Frame", padding="10")
        search_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        # Video name input
        ttk.Label(search_frame, text="Video Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.video_name_var = tk.StringVar()
        self.video_name_entry = ttk.Entry(search_frame, textvariable=self.video_name_var, width=40)
        self.video_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(search_frame, text="Auto-fill", 
                  command=self.auto_fill_video_name).grid(row=0, column=2)
        
        # Time input
        ttk.Label(search_frame, text="Time (seconds):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.time_var = tk.StringVar()
        self.time_entry = ttk.Entry(search_frame, textvariable=self.time_var, width=20)
        self.time_entry.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        
        # Search button
        ttk.Button(search_frame, text="🔍 Search Frame", 
                  command=self.search_frame).grid(row=1, column=2, pady=(10, 0))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="4. Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=8, width=70, 
                                   font=('Consolas', 10), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                        command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select a video folder")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              foreground='blue', font=('Arial', 9))
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure main frame grid weights
        main_frame.rowconfigure(2, weight=1)
    
    def browse_folder(self):
        """Open folder browser dialog and scan for videos"""
        folder_path = filedialog.askdirectory(title="Select folder containing videos")
        
        if folder_path:
            self.video_folder = folder_path
            self.folder_label.config(text=folder_path, foreground='black')
            self.status_var.set("Scanning videos...")
            
            # Run video scanning in separate thread to avoid GUI freeze
            threading.Thread(target=self.scan_videos, daemon=True).start()
    
    def scan_videos(self):
        """Scan the selected folder for video files and extract metadata"""
        try:
            self.video_info.clear()
            video_files = []
            
            folder_path = Path(self.video_folder)
            
            # Find all video files
            for file_path in folder_path.rglob('*'):
                if file_path.suffix.lower() in self.supported_formats:
                    video_files.append(file_path)
            
            # Update GUI with found videos
            self.root.after(0, lambda: self.update_video_list(video_files))
            
            # Extract metadata for each video
            for i, video_path in enumerate(video_files):
                try:
                    self.root.after(0, lambda i=i, total=len(video_files): 
                                  self.status_var.set(f"Processing video {i+1}/{total}..."))
                    
                    metadata = self.extract_video_metadata(video_path)
                    if metadata:
                        self.video_info[video_path.name] = {
                            'path': str(video_path),
                            'metadata': metadata
                        }
                        
                except Exception as e:
                    print(f"Error processing {video_path.name}: {e}")
                    continue
            
            # Update status
            self.root.after(0, lambda: self.status_var.set(
                f"Found {len(self.video_info)} videos with metadata"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to scan videos: {str(e)}"))
    
    def update_video_list(self, video_files: List[Path]):
        """Update the video listbox with found videos"""
        self.video_listbox.delete(0, tk.END)
        
        for video_path in sorted(video_files):
            display_name = f"📁 {video_path.parent.name} / {video_path.name}"
            self.video_listbox.insert(tk.END, display_name)
    
    def extract_video_metadata(self, video_path: Path) -> Optional[Dict]:
        """Extract metadata from video file using OpenCV"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size': video_path.stat().st_size
            }
            
        except Exception as e:
            print(f"Error extracting metadata from {video_path}: {e}")
            return None
    
    def on_video_select(self, event):
        """Handle video selection from listbox"""
        selection = self.video_listbox.curselection()
        if selection:
            selected_text = self.video_listbox.get(selection[0])
            # Extract video name from display text
            video_name = selected_text.split(" / ")[-1]
            self.video_name_var.set(video_name)
    
    def auto_fill_video_name(self):
        """Auto-fill video name from current selection"""
        selection = self.video_listbox.curselection()
        if selection:
            selected_text = self.video_listbox.get(selection[0])
            video_name = selected_text.split(" / ")[-1]
            self.video_name_var.set(video_name)
        else:
            messagebox.showwarning("No Selection", "Please select a video from the list first")
    
    def search_frame(self):
        """Search for frame index based on video name and time"""
        video_name = self.video_name_var.get().strip()
        time_str = self.time_var.get().strip()
        
        # Validate inputs
        if not video_name:
            messagebox.showerror("Error", "Please enter a video name")
            return
        
        if not time_str:
            messagebox.showerror("Error", "Please enter a time in seconds")
            return
        
        try:
            time_seconds = float(time_str)
            if time_seconds < 0:
                messagebox.showerror("Error", "Time must be non-negative")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for time")
            return
        
        # Find video in our metadata
        if video_name not in self.video_info:
            messagebox.showerror("Error", f"Video '{video_name}' not found in scanned videos")
            return
        
        video_data = self.video_info[video_name]
        metadata = video_data['metadata']
        
        # Calculate frame index
        frame_index = self.calculate_frame_index(time_seconds, metadata)
        
        # Display results
        self.display_results(video_name, time_seconds, frame_index, metadata)
    
    def calculate_frame_index(self, time_seconds: float, metadata: Dict) -> Tuple[int, Dict]:
        """Calculate frame index from time and return additional info"""
        fps = metadata['fps']
        frame_count = metadata['frame_count']
        duration = metadata['duration']
        
        # Calculate frame index (0-based)
        frame_index = int(time_seconds * fps)
        
        # Validate frame index is within bounds
        if frame_index >= frame_count:
            frame_index = frame_count - 1  # Use last frame
            actual_time = frame_index / fps
        else:
            actual_time = frame_index / fps
        
        return frame_index, {
            'actual_time': actual_time,
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count,
            'is_clamped': frame_index >= frame_count - 1
        }
    
    def display_results(self, video_name: str, requested_time: float, 
                       result: Tuple[int, Dict], metadata: Dict):
        """Display search results in the results text area"""
        frame_index, calc_info = result
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        results = []
        results.append("=" * 60)
        results.append("🎬 FRAME SEARCH RESULTS")
        results.append("=" * 60)
        results.append("")
        results.append(f"📹 Video: {video_name}")
        results.append(f"📂 Path: {self.video_info[video_name]['path']}")
        results.append("")
        results.append("⏰ TIME INFORMATION:")
        results.append(f"   Requested Time: {requested_time:.3f} seconds")
        results.append(f"   Actual Time: {calc_info['actual_time']:.3f} seconds")
        if calc_info['is_clamped']:
            results.append(f"   ⚠️ WARNING: Requested time exceeds video duration!")
        results.append("")
        results.append("🎯 FRAME INDEX RESULT:")
        results.append(f"   Frame Index: {frame_index}")
        results.append(f"   Frame Count: {calc_info['frame_count']}")
        results.append(f"   Progress: {frame_index/calc_info['frame_count']*100:.2f}%")
        results.append("")
        results.append("📊 VIDEO METADATA:")
        results.append(f"   FPS: {calc_info['fps']:.2f}")
        results.append(f"   Duration: {calc_info['duration']:.2f} seconds")
        results.append(f"   Resolution: {metadata['width']}x{metadata['height']}")
        results.append(f"   File Size: {metadata['file_size']/1024/1024:.2f} MB")
        results.append("")
        results.append("💡 USAGE:")
        results.append(f"   Use frame_index = {frame_index} in your retrieval system")
        results.append(f"   This corresponds to {calc_info['actual_time']:.3f}s in the video")
        results.append("=" * 60)
        
        # Display results
        results_text = "\n".join(results)
        self.results_text.insert(1.0, results_text)
        
        # Update status
        self.status_var.set(f"Found: frame_index = {frame_index} at {calc_info['actual_time']:.3f}s")
        
        # Also copy result to clipboard for convenience
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(f"frame_index = {frame_index}")
            self.status_var.set(f"Result copied to clipboard! frame_index = {frame_index}")
        except:
            pass
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the video frame search tool"""
    try:
        app = VideoFrameSearchTool()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()
