"""
Agent Tools - Tools for the Unified Conversational Agent
========================================================

Provides tools for the unified Agno agent to interact with the retrieval system,
search keyframes, analyze images, and provide conversational assistance.

Author: Enhanced Retrieval System
Version: 2.0
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import time
from pydantic import BaseModel, Field, field_validator

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agno.tools import tool
    HAS_AGNO = True
except ImportError:
    # Fallback decorator if Agno is not available
    def tool(name: str = None, description: str = None):
        def decorator(func):
            func._tool_name = name or func.__name__
            func._tool_description = description or func.__doc__ or "No description"
            return func
        return decorator
    HAS_AGNO = False

from system import EnhancedRetrievalSystem, SearchOptions


class SearchKeyframesInput(BaseModel):
    """Input model for search_keyframes tool"""
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(default=10, description="Maximum number of results to return", le=50, ge=1)
    
    @classmethod
    def from_string_or_dict(cls, data):
        """Parse from string representation or dict"""
        if isinstance(data, str):
            try:
                # Try to parse as JSON string
                import json
                parsed = json.loads(data.replace("'", '"'))  # Handle single quotes
                return cls(**parsed)
            except:
                # Fallback - treat as query
                return cls(query=data, limit=10)
        elif isinstance(data, dict):
            return cls(**data)
        else:
            return cls(**data)


class AnalyzeKeyframesInput(BaseModel):
    """Input model for analyze_keyframes tool"""
    frame_identifiers: List[str] = Field(..., description="List of frame identifiers in format 'folder_name:frame_id'")
    
    @classmethod
    def from_string_or_dict(cls, data):
        """Parse from string representation or dict"""
        if isinstance(data, str):
            try:
                import json
                parsed = json.loads(data.replace("'", '"'))
                return cls(**parsed)
            except:
                # Fallback - split by comma
                return cls(frame_identifiers=[data] if data else [])
        elif isinstance(data, dict):
            return cls(**data)
        else:
            return cls(**data)


class ChatFramesInput(BaseModel):
    """Input model for chat_about_frames tool"""
    message: str = Field(..., description="User's message or question")
    selected_frames: List[str] = Field(default_factory=list, description="List of selected frame identifiers")
    
    @field_validator('selected_frames', mode='before')
    @classmethod
    def parse_selected_frames(cls, v):
        """Parse selected_frames from various formats"""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Handle string representations
            v = v.strip()
            if v in ['[]', '', 'null', 'none', 'None']:
                return []
            # Try to parse JSON
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except:
                # Split by comma if it looks like a comma-separated list
                if ',' in v:
                    return [item.strip() for item in v.split(',') if item.strip()]
                # Single item
                return [v] if v else []
        return []
    
    @classmethod
    def from_string_or_dict(cls, data):
        """Parse from string representation or dict"""
        if isinstance(data, str):
            try:
                import json
                parsed = json.loads(data.replace("'", '"'))
                return cls(**parsed)
            except:
                # Fallback - treat as message
                return cls(message=data, selected_frames=[])
        elif isinstance(data, dict):
            return cls(**data)
        else:
            return cls(**data)


def get_agent_tools(retrieval_system: EnhancedRetrievalSystem) -> List[Any]:
    """
    Get all available tools for the unified conversational agent
    
    Args:
        retrieval_system: The main retrieval system instance
        
    Returns:
        List of tool functions for the agent
    """
    
    @tool(
        name="search_keyframes",
        description="Search for keyframes using natural language queries. Use this when users want to find specific scenes, objects, actions, or visual content."
    )
    def search_keyframes(query: str, limit: int = 10) -> str:
        """
        Search for keyframes matching a query
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return (default 10)
            
        Returns:
            String describing the search results
        """
        try:
            
            # Create search options
            options = SearchOptions(
                mode="hybrid",  # Use best available search mode
                limit=min(limit, 50),  # Cap at 50 results
                include_temporal_context=False,
                include_explanations=False
            )
            
            # Perform search
            results = retrieval_system.search(query, options)
            
            if not results:
                return f"🔍 No keyframes found for query: '{query}'. Try using different keywords or broader terms."
            
            # Format results for response
            response_lines = [f"🔍 I found {len(results)} keyframes for '{query}':"]
            
            for i, result in enumerate(results[:limit], 1):
                folder = result.metadata.folder_name
                frame = result.metadata.frame_id
                score = result.similarity_score
                
                response_lines.append(
                    f"{i}. Video {folder}, Frame {frame} (similarity: {score:.3f})"
                )
            
            if len(results) > limit:
                response_lines.append(f"... and {len(results) - limit} more results")
            
            return "\n".join(response_lines)
            
        except Exception as e:
            return f"❌ Search failed: {str(e)}"
    
    @tool(
        name="analyze_keyframes", 
        description="Provide detailed analysis of specific keyframes by their identifiers. Use this to give more information about search results."
    )
    def analyze_keyframes(frame_identifiers: str) -> str:
        """
        Analyze specific keyframes in detail
        
        Args:
            frame_identifiers: Comma-separated frame identifiers in format 'folder:frame_id'
            
        Returns:
            String with detailed analysis of the frames
        """
        try:
            # Parse frame identifiers
            if isinstance(frame_identifiers, str):
                frame_list = [f.strip() for f in frame_identifiers.split(',') if f.strip()]
            else:
                frame_list = frame_identifiers
            
            if not frame_list:
                return "❌ No frame identifiers provided for analysis."
            
            analyses = []
            
            for identifier in frame_list[:5]:  # Limit to 5 frames
                try:
                    if ":" not in identifier:
                        analyses.append(f"❌ Invalid identifier format: '{identifier}'. Use 'folder:frame_id'")
                        continue
                    
                    folder_name, frame_id = identifier.split(":", 1)
                    
                    # Find the keyframe metadata
                    # This is a simplified implementation - you might need to access the index directly
                    analysis = f"📊 Analysis of {folder_name}:{frame_id}:\n"
                    analysis += f"   - Video: {folder_name}\n"
                    analysis += f"   - Frame ID: {frame_id}\n"
                    analysis += f"   - This appears to be a keyframe from video {folder_name}"
                    
                    analyses.append(analysis)
                    
                except Exception as e:
                    analyses.append(f"❌ Error analyzing {identifier}: {str(e)}")
            
            return "\n\n".join(analyses)
            
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    @tool(
        name="get_system_status",
        description="Get current status and statistics of the retrieval system. Use this when users ask about system health, performance, or capabilities."
    )
    def get_system_status() -> str:
        """
        Get system status and statistics
        
        Returns:
            String with current system status and key metrics
        """
        try:
            status = retrieval_system.status
            stats = retrieval_system.get_system_stats()
            
            response_lines = ["📊 System Status:"]
            response_lines.append(f"   • Status: {'🟢 Ready' if status.is_ready else '🔴 Not Ready'}")
            response_lines.append(f"   • Index: {'✅ Loaded' if status.index_loaded else '❌ Not Loaded'}")
            
            # Add key statistics
            if stats:
                index_stats = stats.get("index", {})
                if index_stats:
                    response_lines.append(f"   • Total Keyframes: {index_stats.get('total_keyframes', 'Unknown')}")
                    response_lines.append(f"   • Video Folders: {index_stats.get('total_folders', 'Unknown')}")
                
                performance_stats = stats.get("performance", {})
                if performance_stats and "operations" in performance_stats:
                    search_ops = performance_stats["operations"].get("search_query", {})
                    if search_ops:
                        avg_time = search_ops.get("avg_time", 0)
                        response_lines.append(f"   • Avg Search Time: {avg_time:.3f}s")
            
            return "\n".join(response_lines)
            
        except Exception as e:
            return f"❌ Failed to get system status: {str(e)}"
    
    @tool(
        name="chat_about_frames",
        description="Have a conversation about selected keyframes or provide general help. Use this for open-ended questions or when users want to chat about the system."
    )
    def chat_about_frames(message: str, selected_frames: str = "") -> str:
        """
        General chat function about keyframes or the system
        
        Args:
            message: User's message or question
            selected_frames: Comma-separated list of selected frame identifiers (optional)
            
        Returns:
            Conversational response about the frames or system
        """
        try:
            # Parse selected frames
            frame_list = []
            if selected_frames:
                frame_list = [f.strip() for f in selected_frames.split(',') if f.strip()]
            
            response_lines = []
            
            # Handle different types of messages
            message_lower = message.lower()
            
            if "help" in message_lower:
                response_lines.extend([
                    "🤖 I'm your AI assistant for the video keyframe retrieval system!",
                    "",
                    "I can help you:",
                    "• 🔍 Search for keyframes using natural language",
                    "• 📊 Analyze specific keyframes in detail", 
                    "• 📈 Check system status and performance",
                    "• 💬 Answer questions about the system",
                    "",
                    "Try asking me things like:",
                    "• 'Find scenes with people talking'",
                    "• 'Show me outdoor car scenes'",
                    "• 'What's the system status?'",
                    "• 'Analyze video L01_V001 frame 25'"
                ])
            
            elif "thank" in message_lower:
                response_lines.append("😊 You're welcome! Happy to help with your keyframe searches!")
            
            elif frame_list:
                count = len(frame_list)
                response_lines.extend([
                    f"🖼️ I can see you have {count} frame{'s' if count != 1 else ''} selected:",
                    ""
                ])
                
                for frame in frame_list[:5]:  # Show up to 5
                    response_lines.append(f"   • {frame}")
                
                if len(frame_list) > 5:
                    response_lines.append(f"   ... and {len(frame_list) - 5} more")
                
                response_lines.extend([
                    "",
                    "Feel free to ask me questions about these frames or search for similar content!"
                ])
            
            else:
                # General conversational response
                response_lines.extend([
                    f"💬 I understand you want to chat about: '{message}'",
                    "",
                    "I'm here to help with keyframe searching and analysis. You can:",
                    "• Search for specific scenes or objects",
                    "• Get detailed information about keyframes", 
                    "• Check system performance",
                    "",
                    "What would you like to explore in the keyframe database?"
                ])
            
            return "\n".join(response_lines)
            
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    # Return list of all tool functions
    tools = [
        search_keyframes,
        analyze_keyframes, 
        get_system_status,
        chat_about_frames
    ]
    
    return tools


def get_fallback_tools(retrieval_system: EnhancedRetrievalSystem) -> Dict[str, Any]:
    """
    Get fallback tools as a dictionary when Agno is not available
    
    Args:
        retrieval_system: The main retrieval system instance
        
    Returns:
        Dictionary mapping tool names to functions
    """
    tools = get_agent_tools(retrieval_system)
    
    fallback_tools = {}
    for tool_func in tools:
        if hasattr(tool_func, '_tool_name'):
            fallback_tools[tool_func._tool_name] = tool_func
        else:
            fallback_tools[tool_func.__name__] = tool_func
    
    return fallback_tools


# Export for compatibility
__all__ = ['get_agent_tools', 'get_fallback_tools']


if __name__ == "__main__":
    print("Agent Tools Module")
    print("=" * 30)
    
    if HAS_AGNO:
        print("✅ Agno available - tools ready for agent integration")
    else:
        print("⚠️ Agno not available - using fallback tool definitions")
    
    # Show available tools
    print("\nAvailable tools:")
    tools = get_agent_tools(None)  # Pass None for demonstration
    for i, tool_func in enumerate(tools, 1):
        tool_name = getattr(tool_func, '_tool_name', tool_func.__name__)
        tool_desc = getattr(tool_func, '_tool_description', 'No description')
        print(f"{i}. {tool_name}: {tool_desc}")