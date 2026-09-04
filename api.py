"""
Enhanced Retrieval System - API & Communication Layer
====================================================

Complete server and API functionality including:
- Real-time SocketIO communication server
- REST API endpoints for modern integration
- Request handling and validation
- Translation services integration
- Rate limiting and security features

Author: Enhanced Retrieval System
Version: 2.0
"""

import time
import threading
import socket
import json
import os
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid
import asyncio
from datetime import datetime, timedelta

# Web framework imports
try:
    import socketio
    import eventlet
    import eventlet.wsgi
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Translation service
try:
    from googletrans import Translator
    import spacy
    HAS_TRANSLATION = True
except ImportError:
    HAS_TRANSLATION = False

from utils import Config, Logger, PerformanceMonitor, CacheManager
from core import SearchResult, KeyframeMetadata
from system import EnhancedRetrievalSystem, SearchOptions


class RemoteUnifiedIndex:
    """
    🌐 Remote Unified Index - Network-based index access
    
    Provides the same interface as local unified index but operates over HTTP.
    Allows transparent access to remote indexes without local storage.
    """
    
    def __init__(self, host: str, port: int, index_id: str, network_client):
        self.host = host
        self.port = port
        self.index_id = index_id
        self.network_client = network_client
        self.base_url = f"http://{host}:{port}/api/indexes/{index_id}"
        
        # Cache for performance
        self._info_cache = None
        self._metadata_cache = {}
        
    def get_index_info(self) -> Dict[str, Any]:
        """Get index information from remote server"""
        if self._info_cache:
            return self._info_cache
            
        try:
            response = self.network_client.session.get(f"{self.base_url}/info")
            if response.status_code == 200:
                self._info_cache = response.json()
                return self._info_cache
            else:
                raise Exception(f"Failed to get index info: HTTP {response.status_code}")
        except Exception as e:
            raise Exception(f"Remote index info error: {e}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector search on remote index"""
        try:
            # Send search request to remote server
            search_data = {
                'query_vector': query_vector.tolist(),
                'k': k
            }
            
            response = self.network_client.session.post(
                f"{self.base_url}/search", 
                json=search_data
            )
            
            if response.status_code == 200:
                results = response.json()
                return results.get('results', [])
            else:
                raise Exception(f"Search failed: HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Remote search error: {e}")
    
    def get_metadata(self, frame_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific frames from remote index"""
        try:
            response = self.network_client.session.post(
                f"{self.base_url}/metadata",
                json={'frame_ids': frame_ids}
            )
            
            if response.status_code == 200:
                return response.json().get('metadata', [])
            else:
                raise Exception(f"Metadata fetch failed: HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Remote metadata error: {e}")
    
    @property
    def ntotal(self) -> int:
        """Get total number of vectors in remote index"""
        try:
            info = self.get_index_info()
            return info.get('total_vectors', 0)
        except:
            return 0
    
    def get_full_image_fast(self, frame_index: int) -> bytes:
        """Get full image bytes for a specific frame from remote index"""
        try:
            response = self.network_client.session.get(
                f"{self.base_url}/image/{frame_index}"
            )
            
            if response.status_code == 200:
                return response.content
            else:
                return None
                
        except Exception as e:
            if hasattr(self.network_client, 'logger') and self.network_client.logger:
                self.network_client.logger.warning(f"Failed to get remote image {frame_index}: {e}")
            return None
    
    def get_thumbnail_fast(self, frame_index: int) -> np.ndarray:
        """Get thumbnail array for a specific frame from remote index"""
        try:
            response = self.network_client.session.get(
                f"{self.base_url}/thumbnail/{frame_index}"
            )
            
            if response.status_code == 200:
                # Convert JPEG bytes back to numpy array
                import cv2
                import numpy as np
                
                # Decode JPEG bytes
                nparr = np.frombuffer(response.content, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    # Convert BGR back to RGB
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    return img_rgb
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            if hasattr(self.network_client, 'logger') and self.network_client.logger:
                self.network_client.logger.warning(f"Failed to get remote thumbnail {frame_index}: {e}")
            return None
    
    @property
    def unified_index(self):
        """Property to make RemoteUnifiedIndex compatible with GUI thumbnail code"""
        return self
    
    @property 
    def is_loaded(self) -> bool:
        """Check if remote index is available"""
        try:
            info = self.get_index_info()
            return info is not None
        except:
            return False
    
    def get_full_image(self, frame_index: int) -> bytes:
        """Get full image bytes for a specific frame (alias for compatibility)"""
        return self.get_full_image_fast(frame_index)
    
    def get_thumbnail(self, frame_index: int) -> np.ndarray:
        """Get thumbnail array for a specific frame (alias for compatibility)"""
        return self.get_thumbnail_fast(frame_index)
    
    def search_unified_fast(self, query_vector: np.ndarray, k: int = 10, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform unified search on remote index - compatible with local unified index interface
        """
        try:
            # Use the existing search method but ensure results are in unified format
            results = self.search(query_vector, k)
            
            # Convert results to unified format matching local unified index
            unified_results = []
            for result in results:
                if isinstance(result, dict):
                    # Ensure all required fields are present
                    unified_result = {
                        'metadata': result.get('metadata', {}),
                        'similarity_score': result.get('similarity_score', 0.0),
                        'rank': result.get('rank', len(unified_results) + 1),
                        'index': result.get('index', len(unified_results))  # This will be used as unified_index
                    }
                    
                    # Apply similarity threshold filter
                    if unified_result['similarity_score'] >= similarity_threshold:
                        unified_results.append(unified_result)
            
            return unified_results
            
        except Exception as e:
            if hasattr(self.network_client, 'logger') and self.network_client.logger:
                self.network_client.logger.error(f"Remote unified search failed: {e}")
            raise Exception(f"Remote unified search error: {e}")


# ============================================
# DISTRIBUTED NETWORK SYSTEM COMPONENTS  
# ============================================

@dataclass
class NetworkNode:
    """Network node information"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    shared_folders: List[str]
    available_indexes: List[str]
    last_seen: datetime
    node_type: str = "peer"  # peer, server, client
    status: str = "active"   # active, inactive, unreachable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'port': self.port,
            'shared_folders': self.shared_folders,
            'available_indexes': self.available_indexes,
            'last_seen': self.last_seen.isoformat(),
            'node_type': self.node_type,
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkNode':
        # Convert last_seen string back to datetime with error handling
        try:
            if isinstance(data['last_seen'], str):
                data['last_seen'] = datetime.fromisoformat(data['last_seen'])
            elif not isinstance(data['last_seen'], datetime):
                data['last_seen'] = datetime.now()
        except (ValueError, KeyError):
            data['last_seen'] = datetime.now()
        return cls(**data)


@dataclass
class SharedFolder:
    """Shared folder configuration"""
    folder_id: str
    path: str
    name: str
    permissions: List[str]  # ["read", "write", "execute"]
    description: str = ""
    file_count: int = 0
    total_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedFolder':
        return cls(**data)


class NetworkDiscovery:
    """
    🔍 Network Discovery Service
    
    Auto-discovers peers on LAN using UDP broadcast
    Maintains active node registry
    """
    
    def __init__(self, port: int = 5000, logger=None):
        self.port = port
        self.logger = logger or Logger()
        self.broadcast_port = port + 1
        # Also listen on legacy broadcast port for backward compatibility
        self.legacy_broadcast_port = 5556
        
        # Node management
        self.local_node = self._create_local_node()
        self.discovered_nodes: Dict[str, NetworkNode] = {}
        self.discovery_active = False
        
        # Network sockets
        self.discovery_socket = None
        self.broadcast_socket = None
        
        # Threading
        self.discovery_thread = None
        self.broadcast_thread = None
        self.cleanup_thread = None
        
        # Config - Increased timeouts for better network discovery
        self.discovery_interval = 15  # seconds - More frequent broadcasts
        self.node_timeout = 300      # seconds - 5 minutes before marking inactive
        
    def _create_local_node(self) -> NetworkNode:
        """Create local node information"""
        hostname = socket.gethostname()
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"
            
        node_id = hashlib.md5(f"{hostname}_{local_ip}".encode()).hexdigest()[:12]
        
        return NetworkNode(
            node_id=node_id,
            hostname=hostname,
            ip_address=local_ip,
            port=self.port,
            shared_folders=[],
            available_indexes=[],
            last_seen=datetime.now(),
            node_type="peer"
        )
    
    def start_discovery(self):
        """Start network discovery service"""
        if self.discovery_active:
            return
            
        try:
            self.discovery_active = True
            
            # Start UDP listener for discovery messages on new port
            self.discovery_thread = threading.Thread(target=self._discovery_listener, daemon=True)
            self.discovery_thread.start()
            
            # Start legacy UDP listener for backward compatibility
            self.legacy_discovery_thread = threading.Thread(target=self._legacy_discovery_listener, daemon=True)
            self.legacy_discovery_thread.start()
            
            # Start periodic broadcast
            self.broadcast_thread = threading.Thread(target=self._periodic_broadcast, daemon=True)
            self.broadcast_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_inactive_nodes, daemon=True)
            self.cleanup_thread.start()
            
            self.logger.info(f"🔍 Network discovery started on port {self.broadcast_port}")
            self.logger.info(f"📡 Local node: {self.local_node.hostname} ({self.local_node.ip_address})")
            
        except Exception as e:
            self.logger.error(f"Failed to start network discovery: {e}")
            self.discovery_active = False
    
    def stop_discovery(self):
        """Stop network discovery service"""
        self.discovery_active = False
        
        if self.discovery_socket:
            self.discovery_socket.close()
        if hasattr(self, 'legacy_discovery_socket') and self.legacy_discovery_socket:
            self.legacy_discovery_socket.close()
        if self.broadcast_socket:
            self.broadcast_socket.close()
            
        self.logger.info("🔍 Network discovery stopped")
    
    def _discovery_listener(self):
        """Listen for discovery broadcasts"""
        try:
            self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.discovery_socket.bind(('', self.broadcast_port))
            
            while self.discovery_active:
                try:
                    data, addr = self.discovery_socket.recvfrom(8192)
                    self._handle_discovery_message(data, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.discovery_active:
                        self.logger.warning(f"Discovery listener error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to start discovery listener: {e}")
    
    def _legacy_discovery_listener(self):
        """Listen for legacy discovery broadcasts on port 5556"""
        try:
            self.legacy_discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.legacy_discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.legacy_discovery_socket.bind(('', self.legacy_broadcast_port))
            
            self.logger.info(f"🔍 Legacy discovery listener started on port {self.legacy_broadcast_port}")
            
            while self.discovery_active:
                try:
                    data, addr = self.legacy_discovery_socket.recvfrom(8192)
                    self.logger.info(f"📡 Legacy broadcast received from {addr[0]}:{addr[1]}")
                    self._handle_discovery_message(data, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.discovery_active:
                        self.logger.warning(f"Legacy discovery listener error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to start legacy discovery listener: {e}")
    
    def _handle_discovery_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming discovery message"""
        try:
            message = json.loads(data.decode('utf-8'))
            
            if message.get('type') == 'node_announcement':
                node_data = message.get('node')
                if node_data and node_data['node_id'] != self.local_node.node_id:
                    
                    # Update IP address to actual sender IP
                    node_data['ip_address'] = addr[0]
                    node_data['last_seen'] = datetime.now().isoformat()
                    
                    node = NetworkNode.from_dict(node_data)
                    self.discovered_nodes[node.node_id] = node
                    
                    self.logger.info(f"🔍 Discovered node: {node.hostname} ({node.ip_address})")
                    
                    # If this is a discovery request, respond immediately
                    if message.get('discovery_request'):
                        self._trigger_manual_discovery()
                    
        except Exception as e:
            self.logger.warning(f"Failed to handle discovery message: {e}")
    
    def _periodic_broadcast(self):
        """Periodically broadcast node information"""
        try:
            self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while self.discovery_active:
                try:
                    # Update local node info
                    self.local_node.last_seen = datetime.now()
                    
                    message = {
                        'type': 'node_announcement',
                        'node': self.local_node.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    data = json.dumps(message).encode('utf-8')
                    # Broadcast on both new and legacy ports for compatibility
                    self.broadcast_socket.sendto(data, ('<broadcast>', self.broadcast_port))
                    self.broadcast_socket.sendto(data, ('<broadcast>', self.legacy_broadcast_port))
                    
                    time.sleep(self.discovery_interval)
                    
                except Exception as e:
                    if self.discovery_active:
                        self.logger.warning(f"Broadcast error: {e}")
                        time.sleep(5)  # Wait before retry
                        
        except Exception as e:
            self.logger.error(f"Failed to start periodic broadcast: {e}")
    
    def _trigger_manual_discovery(self):
        """Trigger immediate discovery broadcast to force network scan"""
        try:
            if not self.discovery_active or not hasattr(self, 'broadcast_socket') or not self.broadcast_socket:
                return
                
            # Update local node info
            self.local_node.last_seen = datetime.now()
            
            message = {
                'type': 'node_announcement',
                'node': self.local_node.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'discovery_request': True  # Flag to indicate this is a discovery request
            }
            
            data = json.dumps(message).encode('utf-8')
            # Broadcast on both new and legacy ports for compatibility  
            self.broadcast_socket.sendto(data, ('<broadcast>', self.broadcast_port))
            self.broadcast_socket.sendto(data, ('<broadcast>', self.legacy_broadcast_port))
            
            if self.logger:
                self.logger.info(f"🔍 Manual discovery broadcast sent to network")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Manual discovery broadcast failed: {e}")
    
    def _cleanup_inactive_nodes(self):
        """Remove inactive nodes from registry"""
        while self.discovery_active:
            try:
                current_time = datetime.now()
                timeout_threshold = timedelta(seconds=self.node_timeout)
                
                inactive_nodes = []
                for node_id, node in self.discovered_nodes.items():
                    if current_time - node.last_seen > timeout_threshold:
                        inactive_nodes.append(node_id)
                
                for node_id in inactive_nodes:
                    removed_node = self.discovered_nodes.pop(node_id)
                    self.logger.info(f"🔍 Removed inactive node: {removed_node.hostname}")
                
                time.sleep(120)  # Check every 2 minutes - less frequent cleanup
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                time.sleep(60)
    
    def get_discovered_nodes(self) -> List[NetworkNode]:
        """Get list of discovered active nodes"""
        return list(self.discovered_nodes.values())
    
    def update_shared_folders(self, folders: List[SharedFolder]):
        """Update local node's shared folders"""
        self.local_node.shared_folders = [f.name for f in folders]
    
    def update_available_indexes(self, indexes: List[str]):
        """Update local node's available indexes"""
        self.local_node.available_indexes = indexes
    
    def discover_nodes(self, timeout: float = 10.0) -> List[NetworkNode]:
        """Discover nodes on network with timeout - maintains persistent discovery"""
        try:
            # Ensure discovery is running persistently
            if not self.discovery_active:
                self.start_discovery()
                time.sleep(1.0)  # Brief wait for service to start
            
            # Trigger manual discovery broadcast to force immediate scan
            self._trigger_manual_discovery()
            
            # Wait for responses from network nodes
            time.sleep(timeout)
            
            # Return discovered nodes (don't stop discovery service)
            return self.get_discovered_nodes()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Discovery failed: {e}")
            return []
    
    def _ping_node(self, host: str, port: int) -> Optional[Dict]:
        """Ping a specific node to check if it's alive"""
        try:
            import requests
            response = requests.get(f"http://{host}:{port}/api/ping", timeout=2)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Ping failed for {host}:{port}: {e}")
            return None


class NetworkServer:
    """
    🌐 Distributed Retrieval Network Server
    
    HTTP server for sharing folders and .rvdb files across network
    Provides REST API for remote resource access
    """
    
    def __init__(self, port: int = 5000, logger=None, retrieval_system=None):
        self.port = port
        self.logger = logger or Logger()
        self.retrieval_system = retrieval_system
        
        # Server state
        self.server_active = False
        self.is_running = False  # Alias for backward compatibility
        self.server_thread = None  # Track Flask server thread
        self.start_time = None  # Track server start time
        self.shared_folders: Dict[str, SharedFolder] = {}
        self.available_indexes: Dict[str, str] = {}  # {index_id: file_path}
        self.active_connections: Dict[str, Dict] = {}
        
        # Security
        self.auth_tokens: Dict[str, Dict] = {}  # {token: {user, expires, permissions}}
        self.rate_limiter = RateLimiter(max_requests=100, window_minutes=1)
        
        # Flask app
        if HAS_FLASK:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()
        else:
            self.app = None
            
        # Network discovery integration
        self.discovery = NetworkDiscovery(port=port, logger=logger)
        
    def _setup_routes(self):
        """Setup Flask REST API routes"""
        
        @self.app.route('/api/ping', methods=['GET'])
        def ping():
            """Health check endpoint"""
            return jsonify({
                'status': 'active',
                'node_id': self.discovery.local_node.node_id,
                'hostname': self.discovery.local_node.hostname,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/node/info', methods=['GET'])
        def node_info():
            """Get detailed node information"""
            return jsonify({
                'node_id': self.discovery.local_node.node_id,
                'hostname': self.discovery.local_node.hostname,
                'ip_address': self.discovery.local_node.ip_address,
                'port': self.port,
                'status': 'active',
                'server_active': self.server_active,
                'shared_folders': len(self.shared_folders),
                'available_indexes': len(self.available_indexes),
                'active_connections': len(self.active_connections),
                'discovery_active': self.discovery.discovery_active,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/status', methods=['GET'])
        def server_status():
            """Get server status information"""
            uptime_seconds = 0
            if self.start_time:
                uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
            return jsonify({
                'server_active': self.server_active,
                'is_running': self.is_running,
                'port': self.port,
                'discovery_active': self.discovery.discovery_active,
                'uptime_seconds': uptime_seconds,
                'thread_alive': self.server_thread.is_alive() if self.server_thread else False,
                'version': '3.0',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/folders', methods=['GET'])
        def list_folders():
            """List available shared folders"""
            try:
                folders = [folder.to_dict() for folder in self.shared_folders.values()]
                return jsonify({'folders': folders})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/folders/<folder_id>/files', methods=['GET'])
        def list_files(folder_id):
            """List files in shared folder"""
            try:
                if folder_id not in self.shared_folders:
                    return jsonify({'error': 'Folder not found'}), 404
                
                folder = self.shared_folders[folder_id]
                folder_path = Path(folder.path)
                
                if not folder_path.exists():
                    return jsonify({'error': 'Folder path does not exist'}), 404
                
                files = []
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(folder_path)
                        files.append({
                            'name': file_path.name,
                            'path': str(relative_path),
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                
                return jsonify({
                    'folder': folder.to_dict(),
                    'files': files,
                    'total_files': len(files)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/folders/<folder_id>/download/<path:file_path>', methods=['GET'])
        def download_file(folder_id, file_path):
            """Download file from shared folder"""
            try:
                if folder_id not in self.shared_folders:
                    return jsonify({'error': 'Folder not found'}), 404
                
                folder = self.shared_folders[folder_id]
                full_path = Path(folder.path) / file_path
                
                if not full_path.exists() or not full_path.is_file():
                    return jsonify({'error': 'File not found'}), 404
                
                # Security: ensure file is within shared folder
                if not str(full_path.resolve()).startswith(str(Path(folder.path).resolve())):
                    return jsonify({'error': 'Access denied'}), 403
                
                from flask import send_file
                return send_file(str(full_path), as_attachment=True)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes', methods=['GET'])
        def list_indexes():
            """List available .rvdb indexes"""
            try:
                indexes = []
                for index_id, file_path in self.available_indexes.items():
                    path_obj = Path(file_path)
                    if path_obj.exists():
                        indexes.append({
                            'index_id': index_id,
                            'name': path_obj.name,
                            'path': str(path_obj),
                            'size': path_obj.stat().st_size,
                            'modified': datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
                        })
                
                return jsonify({'indexes': indexes})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/info', methods=['GET'])
        def get_index_info(index_id):
            """Get detailed information about an index"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                file_path = self.available_indexes[index_id]
                
                # Try to load index info using unified index
                try:
                    from unified_index import UnifiedIndex
                    unified_index = UnifiedIndex(logger=self.logger)
                    unified_index.load_unified_index(file_path)
                    
                    build_info = unified_index._load_build_metadata()
                    
                    return jsonify({
                        'index_id': index_id,
                        'file_path': file_path,
                        'build_info': build_info,
                        'status': 'loaded'
                    })
                    
                except Exception as e:
                    return jsonify({
                        'index_id': index_id,
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed_to_load'
                    })
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/download', methods=['GET'])
        def download_index(index_id):
            """Download .rvdb index file"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                file_path = self.available_indexes[index_id]
                if not Path(file_path).exists():
                    return jsonify({'error': 'Index file not found'}), 404
                
                from flask import send_file
                return send_file(file_path, as_attachment=True)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/search', methods=['POST'])
        def remote_search():
            """Perform search on remote server"""
            try:
                data = request.json
                query = data.get('query')
                options = data.get('options', {})
                
                if not query:
                    return jsonify({'error': 'Query required'}), 400
                
                if not self.retrieval_system:
                    return jsonify({'error': 'Retrieval system not available'}), 503
                
                # Perform search
                from system import SearchOptions
                search_options = SearchOptions(
                    mode=options.get('mode', 'hybrid'),
                    limit=options.get('limit', 50),
                    include_temporal_context=options.get('include_temporal_context', True),
                    include_explanations=options.get('include_explanations', False)
                )
                
                results = self.retrieval_system.search(query, search_options)
                
                # Convert results to JSON-serializable format
                search_results = []
                for result in results:
                    search_results.append({
                        'metadata': {
                            'folder_name': result.metadata.folder_name,
                            'image_name': result.metadata.image_name,
                            'frame_id': result.metadata.frame_id,
                            'file_path': result.metadata.file_path
                        },
                        'similarity_score': result.similarity_score,
                        'rank': result.rank
                    })
                
                return jsonify({
                    'query': query,
                    'results': search_results,
                    'total_results': len(search_results),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/search', methods=['POST'])
        def search_remote_index(index_id):
            """Search in specific remote index"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                data = request.json
                if not data:
                    return jsonify({'error': 'Request body required'}), 400
                
                query_vector = data.get('query_vector')
                k = data.get('k', 10)
                
                if not query_vector:
                    return jsonify({'error': 'query_vector required'}), 400
                
                # Load the unified index
                file_path = self.available_indexes[index_id]
                from unified_index import UnifiedIndex
                unified_index = UnifiedIndex(logger=self.logger)
                unified_index.load_unified_index(file_path)
                
                # Convert query vector to numpy array
                import numpy as np
                query_vector = np.array(query_vector, dtype=np.float32)
                
                # Perform search
                results = unified_index.search_vectors(query_vector, k=k)
                
                # Convert results to JSON serializable format
                search_results = []
                for result in results:
                    search_results.append({
                        'metadata': result['metadata'],
                        'similarity_score': result['similarity_score'],
                        'rank': result['rank'],
                        'index': result['index']
                    })
                
                unified_index.close()
                
                return jsonify({
                    'index_id': index_id,
                    'results': search_results,
                    'total_results': len(search_results),
                    'k': k,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Search failed for index {index_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/metadata', methods=['GET'])
        def get_index_metadata(index_id):
            """Get metadata from specific index"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                # Load the unified index
                file_path = self.available_indexes[index_id]
                from unified_index import UnifiedIndex
                unified_index = UnifiedIndex(logger=self.logger)
                unified_index.load_unified_index(file_path)
                
                # Get metadata list
                metadata_list = unified_index.metadata_list if hasattr(unified_index, 'metadata_list') else []
                
                # Get build info
                build_info = unified_index._load_build_metadata()
                
                unified_index.close()
                
                return jsonify({
                    'index_id': index_id,
                    'metadata_count': len(metadata_list),
                    'build_info': build_info,
                    'sample_metadata': metadata_list[:10] if metadata_list else [],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to get metadata for index {index_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/image/<int:frame_index>', methods=['GET'])
        def get_frame_image(index_id, frame_index):
            """Get full image for a specific frame from unified index"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                file_path = self.available_indexes[index_id]
                from unified_index import UnifiedIndex
                unified_index = UnifiedIndex(logger=self.logger)
                unified_index.load_unified_index(file_path)
                
                # Get full image bytes
                image_bytes = unified_index.get_full_image(frame_index)
                if image_bytes is None:
                    unified_index.close()
                    return jsonify({'error': 'Image not found'}), 404
                
                unified_index.close()
                
                # Return image as binary response
                from flask import Response
                return Response(image_bytes, mimetype='image/jpeg')
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to get frame image: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indexes/<index_id>/thumbnail/<int:frame_index>', methods=['GET'])
        def get_frame_thumbnail(index_id, frame_index):
            """Get thumbnail for a specific frame from unified index"""
            try:
                if index_id not in self.available_indexes:
                    return jsonify({'error': 'Index not found'}), 404
                
                file_path = self.available_indexes[index_id]
                from unified_index import UnifiedIndex
                unified_index = UnifiedIndex(logger=self.logger)
                unified_index.load_unified_index(file_path)
                
                # Get thumbnail array
                thumbnail_array = unified_index.get_thumbnail(frame_index)
                if thumbnail_array is None:
                    unified_index.close()
                    return jsonify({'error': 'Thumbnail not found'}), 404
                
                unified_index.close()
                
                # Convert numpy array to JPEG bytes
                import cv2
                import numpy as np
                
                # Convert RGB to BGR for cv2
                thumbnail_bgr = cv2.cvtColor(thumbnail_array, cv2.COLOR_RGB2BGR)
                
                # Encode to JPEG
                success, encoded_img = cv2.imencode('.jpg', thumbnail_bgr)
                if not success:
                    return jsonify({'error': 'Failed to encode thumbnail'}), 500
                
                # Return as binary response
                from flask import Response
                return Response(encoded_img.tobytes(), mimetype='image/jpeg')
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to get frame thumbnail: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_file():
            """Upload file to server's shared folder"""
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Get target folder (optional, default to first shared folder)
                target_folder_id = request.form.get('folder_id')
                if not target_folder_id:
                    if self.shared_folders:
                        target_folder_id = list(self.shared_folders.keys())[0]
                    else:
                        return jsonify({'error': 'No shared folders available'}), 400
                
                if target_folder_id not in self.shared_folders:
                    return jsonify({'error': f'Folder {target_folder_id} not found'}), 404
                
                # Save file to shared folder
                folder = self.shared_folders[target_folder_id]
                folder_path = Path(folder.path)
                folder_path.mkdir(exist_ok=True, parents=True)
                
                # Secure filename
                import os
                filename = file.filename
                # Remove any path separators for security
                filename = os.path.basename(filename)
                
                save_path = folder_path / filename
                
                # Handle duplicate filenames
                counter = 1
                original_stem = save_path.stem
                original_suffix = save_path.suffix
                while save_path.exists():
                    new_name = f"{original_stem}_{counter}{original_suffix}"
                    save_path = folder_path / new_name
                    counter += 1
                
                # Save the file
                file.save(str(save_path))
                
                if self.logger:
                    self.logger.info(f"📁 File uploaded: {save_path.name} to {folder.name}")
                
                return jsonify({
                    'success': True,
                    'filename': save_path.name,
                    'folder': folder.name,
                    'path': str(save_path.relative_to(folder_path)),
                    'size': save_path.stat().st_size,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Upload failed: {e}")
                return jsonify({'error': str(e)}), 500
    
    def start_server(self):
        """Start network server and discovery"""
        if not HAS_FLASK:
            self.logger.error("Flask not available - cannot start network server")
            return False
        
        # Check if server is already running
        if self.server_active or self.is_running:
            self.logger.warning(f"Network server is already running on port {self.port}")
            return True
            
        # Test port availability
        if not self._test_port_available(self.port):
            self.logger.error(f"Port {self.port} is already in use")
            return False
            
        try:
            self.server_active = True
            self.is_running = True  # Update alias
            
            # Start network discovery
            self.discovery.start_discovery()
            
            # Update discovery with current shared resources
            self.discovery.update_shared_folders(list(self.shared_folders.values()))
            self.discovery.update_available_indexes(list(self.available_indexes.keys()))
            
            # Start Flask server in thread with proper shutdown capability
            import werkzeug.serving
            self.server_thread = threading.Thread(
                target=lambda: self.app.run(
                    host='0.0.0.0', 
                    port=self.port, 
                    debug=False, 
                    use_reloader=False,
                    threaded=True
                ), 
                daemon=True
            )
            self.server_thread.start()
            self.start_time = datetime.now()
            
            # Give Flask a moment to start listening
            time.sleep(0.5)
            
            self.logger.info(f"🌐 Network server started on port {self.port}")
            self.logger.info(f"📡 REST API: http://0.0.0.0:{self.port}/api/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start network server: {e}")
            self.server_active = False
            self.is_running = False  # Update alias
            self.server_thread = None
            self.start_time = None
            return False
    
    def start(self):
        """Alias for start_server() for backward compatibility"""
        return self.start_server()
    
    def stop_server(self):
        """Stop network server and discovery"""
        self.server_active = False
        self.is_running = False  # Update alias
        self.server_thread = None
        self.start_time = None
        self.discovery.stop_discovery()
        self.logger.info("🌐 Network server stopped")
    
    def stop(self):
        """Alias for stop_server() for backward compatibility"""
        return self.stop_server()
    
    def _test_port_available(self, port):
        """Test if port is available for binding"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            sock.close()
            return True
        except:
            return False
    
    def add_shared_folder(self, folder_path: str, name: str = None, 
                         permissions: List[str] = None) -> str:
        """Add folder to shared resources"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder does not exist: {folder_path}")
            
            folder_id = hashlib.md5(str(folder_path).encode()).hexdigest()[:12]
            name = name or folder_path.name
            permissions = permissions or ["read"]
            
            # Calculate folder stats
            file_count = len(list(folder_path.rglob('*')))
            total_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
            
            shared_folder = SharedFolder(
                folder_id=folder_id,
                path=str(folder_path),
                name=name,
                permissions=permissions,
                description=f"Shared folder: {name}",
                file_count=file_count,
                total_size=total_size
            )
            
            self.shared_folders[folder_id] = shared_folder
            
            # Update discovery
            if self.discovery.discovery_active:
                self.discovery.update_shared_folders(list(self.shared_folders.values()))
            
            self.logger.info(f"📁 Added shared folder: {name} ({file_count} files)")
            return folder_id
            
        except Exception as e:
            self.logger.error(f"Failed to add shared folder: {e}")
            raise
    
    def add_shared_index(self, index_path: str) -> str:
        """Add .rvdb index to shared resources"""
        try:
            index_path = Path(index_path)
            if not index_path.exists():
                raise ValueError(f"Index file does not exist: {index_path}")
            
            if not index_path.suffix == '.rvdb':
                raise ValueError("Only .rvdb files are supported")
            
            index_id = hashlib.md5(str(index_path).encode()).hexdigest()[:12]
            self.available_indexes[index_id] = str(index_path)
            
            # Update discovery
            if self.discovery.discovery_active:
                self.discovery.update_available_indexes(list(self.available_indexes.keys()))
            
            self.logger.info(f"📊 Added shared index: {index_path.name}")
            return index_id
            
        except Exception as e:
            self.logger.error(f"Failed to add shared index: {e}")
            raise
    
    def remove_shared_folder(self, folder_id: str):
        """Remove folder from shared resources"""
        if folder_id in self.shared_folders:
            folder = self.shared_folders.pop(folder_id)
            self.logger.info(f"📁 Removed shared folder: {folder.name}")
            
            # Update discovery
            if self.discovery.discovery_active:
                self.discovery.update_shared_folders(list(self.shared_folders.values()))
    
    def remove_shared_index(self, index_id: str):
        """Remove index from shared resources"""
        if index_id in self.available_indexes:
            index_path = self.available_indexes.pop(index_id)
            self.logger.info(f"📊 Removed shared index: {Path(index_path).name}")
            
            # Update discovery
            if self.discovery.discovery_active:
                self.discovery.update_available_indexes(list(self.available_indexes.keys()))
    
    def get_discovered_nodes(self) -> List[NetworkNode]:
        """Get list of discovered network nodes"""
        return self.discovery.get_discovered_nodes()


@dataclass
class APIRequest:
    """Standardized API request structure"""
    endpoint: str
    data: Dict[str, Any]
    client_id: str
    timestamp: float
    request_id: str
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass  
class APIResponse:
    """Standardized API response structure"""
    success: bool
    data: Any = None
    error: str = None
    message: str = None
    request_id: str = None
    timestamp: float = None
    execution_time: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class NetworkClient:
    """
    🔗 Network Client for Remote Resource Access
    
    Connects to remote servers and provides unified access to remote resources
    """
    
    def __init__(self, logger=None):
        self.logger = logger or Logger()
        
        # Connection management
        self.connected_servers: Dict[str, Dict] = {}  # {node_id: connection_info}
        self.cache_manager = CacheManager()
        
        # HTTP session for efficient connections
        try:
            import requests
            self.session = requests.Session()
            self.session.timeout = 30
            self.HAS_REQUESTS = True
        except ImportError:
            self.session = None
            self.HAS_REQUESTS = False
    
    def connect_to_server(self, ip_address: str, port: int = 5000) -> Dict[str, Any]:
        """Connect to remote server and get node info"""
        if not self.HAS_REQUESTS:
            raise ImportError("requests library required for network client")
        
        try:
            # Test connection with ping endpoint
            url = f"http://{ip_address}:{port}/api/ping"
            self.logger.info(f"🔗 Attempting to connect to {url}")
            response = self.session.get(url, timeout=5)  # Shorter timeout
            
            if response.status_code == 200:
                node_info = response.json()
                node_id = node_info['node_id']
                
                # Store connection info
                self.connected_servers[node_id] = {
                    'ip_address': ip_address,
                    'port': port,
                    'hostname': node_info.get('hostname', 'Unknown'),
                    'connected_at': datetime.now(),
                    'last_ping': datetime.now(),
                    'status': 'connected'
                }
                
                self.logger.info(f"🔗 Connected to server: {node_info.get('hostname')} ({ip_address}:{port})")
                return {'success': True, 'node_info': node_info}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"Failed to connect to {ip_address}:{port}: {e}")
            return {'success': False, 'error': str(e)}
    
    def connect(self, ip_address: str, port: int = 5000) -> bool:
        """Alias for connect_to_server() for backward compatibility"""
        result = self.connect_to_server(ip_address, port)
        return result.get('success', False)
    
    def disconnect_server(self, node_id: str):
        """Disconnect from server"""
        if node_id in self.connected_servers:
            server_info = self.connected_servers.pop(node_id)
            self.logger.info(f"🔗 Disconnected from server: {server_info.get('hostname')}")
    
    def get_connected_servers(self) -> List[Dict[str, Any]]:
        """Get list of connected servers"""
        return list(self.connected_servers.values())
    
    def get_shared_folders(self, host: str, port: int) -> List[Dict[str, Any]]:
        """Get shared folders from server by host and port (compatibility method)"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if not node_id:
                self.logger.error(f"No connected server found for {host}:{port}")
                return []
            
            # Get folders using existing method
            result = self.list_remote_folders(node_id)
            if result.get('success'):
                return result.get('folders', [])
            else:
                self.logger.error(f"Failed to get folders: {result.get('error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting shared folders: {e}")
            return []
    
    def get_shared_indexes(self, host: str, port: int) -> List[Dict[str, Any]]:
        """Get shared indexes from server by host and port (compatibility method)"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if not node_id:
                self.logger.error(f"No connected server found for {host}:{port}")
                return []
            
            # Get indexes using HTTP API
            try:
                server_info = self.connected_servers[node_id]
                url = f"http://{server_info['ip_address']}:{server_info['port']}/api/indexes"
                
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('indexes', [])
                else:
                    self.logger.error(f"Failed to get indexes: HTTP {response.status_code}")
                    return []
                    
            except Exception as e:
                self.logger.error(f"HTTP request failed: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting shared indexes: {e}")
            return []
    
    def load_remote_index(self, host: str, port: int, index_id: str) -> Dict[str, Any]:
        """Load remote index and create RemoteUnifiedIndex instance"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if not node_id:
                self.logger.error(f"No connected server found for {host}:{port}")
                return {'success': False, 'error': 'Server not connected'}
            
            # Create RemoteUnifiedIndex instance
            try:
                remote_index = RemoteUnifiedIndex(host, port, index_id, self)
                
                # Test connection and get index info
                index_info = remote_index.get_index_info()
                
                self.logger.info(f"📊 Connected to remote index: {index_id} at {host}:{port}")
                # Extract vector count from build_info
                build_info = index_info.get('build_info', {})
                vector_count = build_info.get('processed_files', 0)
                self.logger.info(f"🔗 Remote index contains {vector_count} vectors")
                
                # Return remote index instance and info for integration
                return {
                    'success': True,
                    'remote_index': remote_index,
                    'index_info': index_info,
                    'connection_url': f"http://{host}:{port}/api/indexes/{index_id}"
                }
                    
            except Exception as e:
                self.logger.error(f"Failed to create remote index: {e}")
                return {'success': False, 'error': str(e)}
                
        except Exception as e:
            self.logger.error(f"Error loading remote index: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_remote_folders(self, node_id: str) -> Dict[str, Any]:
        """List shared folders on remote server"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/folders"
            
            response = self.session.get(url)
            if response.status_code == 200:
                return {'success': True, 'folders': response.json()['folders']}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_remote_files(self, node_id: str, folder_id: str) -> Dict[str, Any]:
        """List files in remote folder"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/folders/{folder_id}/files"
            
            response = self.session.get(url)
            if response.status_code == 200:
                return {'success': True, 'data': response.json()}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def download_remote_file(self, node_id: str, folder_id: str, file_path: str, 
                           local_path: str = None) -> Dict[str, Any]:
        """Download file from remote server"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/folders/{folder_id}/download/{file_path}"
            
            response = self.session.get(url, stream=True)
            if response.status_code == 200:
                
                # Determine local path
                if not local_path:
                    local_path = Path.cwd() / Path(file_path).name
                else:
                    local_path = Path(local_path)
                
                # Create directories if needed
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"📥 Downloaded: {file_path} → {local_path}")
                return {'success': True, 'local_path': str(local_path)}
                
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_remote_indexes(self, node_id: str) -> Dict[str, Any]:
        """List available .rvdb indexes on remote server"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/indexes"
            
            response = self.session.get(url)
            if response.status_code == 200:
                return {'success': True, 'indexes': response.json()['indexes']}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_remote_index_info(self, node_id: str, index_id: str) -> Dict[str, Any]:
        """Get detailed info about remote index"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/indexes/{index_id}/info"
            
            response = self.session.get(url)
            if response.status_code == 200:
                return {'success': True, 'info': response.json()}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def download_remote_index(self, node_id: str, index_id: str, 
                            local_path: str = None) -> Dict[str, Any]:
        """Download .rvdb index from remote server"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/indexes/{index_id}/download"
            
            response = self.session.get(url, stream=True)
            if response.status_code == 200:
                
                # Determine local path
                if not local_path:
                    # Get filename from response headers or use index_id
                    filename = f"remote_index_{index_id}.rvdb"
                    if 'content-disposition' in response.headers:
                        import re
                        cd = response.headers['content-disposition']
                        filename_match = re.search(r'filename="([^"]+)"', cd)
                        if filename_match:
                            filename = filename_match.group(1)
                    
                    local_path = Path.cwd() / filename
                else:
                    local_path = Path(local_path)
                
                # Download index file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"📥 Downloaded index: {index_id} → {local_path}")
                return {'success': True, 'local_path': str(local_path)}
                
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def remote_search(self, node_id: str, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform search on remote server"""
        if node_id not in self.connected_servers:
            return {'success': False, 'error': 'Server not connected'}
        
        try:
            server_info = self.connected_servers[node_id]
            url = f"http://{server_info['ip_address']}:{server_info['port']}/api/search"
            
            payload = {
                'query': query,
                'options': options or {}
            }
            
            response = self.session.post(url, json=payload)
            if response.status_code == 200:
                return {'success': True, 'results': response.json()}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def multi_server_search(self, query: str, options: Dict[str, Any] = None, 
                          server_list: List[str] = None) -> Dict[str, Any]:
        """Search across multiple connected servers"""
        if not server_list:
            server_list = list(self.connected_servers.keys())
        
        results = {}
        total_results = []
        
        for node_id in server_list:
            if node_id in self.connected_servers:
                search_result = self.remote_search(node_id, query, options)
                results[node_id] = search_result
                
                if search_result.get('success'):
                    server_results = search_result['results'].get('results', [])
                    # Add server info to each result
                    for result in server_results:
                        result['source_server'] = {
                            'node_id': node_id,
                            'hostname': self.connected_servers[node_id].get('hostname')
                        }
                    total_results.extend(server_results)
        
        # Sort combined results by ranking score when available, otherwise similarity.
        total_results.sort(key=lambda x: x.get('ranking_score', x.get('similarity_score', 0)), reverse=True)
        
        return {
            'query': query,
            'total_results': len(total_results),
            'results': total_results,
            'server_results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check_servers(self):
        """Check health of all connected servers"""
        inactive_servers = []
        
        for node_id, server_info in self.connected_servers.items():
            try:
                url = f"http://{server_info['ip_address']}:{server_info['port']}/api/ping"
                response = self.session.get(url, timeout=5)
                
                if response.status_code == 200:
                    server_info['last_ping'] = datetime.now()
                    server_info['status'] = 'connected'
                else:
                    server_info['status'] = 'error'
                    inactive_servers.append(node_id)
                    
            except Exception as e:
                server_info['status'] = 'unreachable'
                inactive_servers.append(node_id)
                self.logger.warning(f"Server {node_id} unreachable: {e}")
        
        # Remove inactive servers
        for node_id in inactive_servers:
            self.disconnect_server(node_id)
    
    def search_remote(self, host: str, port: int, query: str, k: int = 50) -> List[Dict[str, Any]]:
        """Search remote server by host and port (compatibility method for GUI)"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if not node_id:
                self.logger.error(f"No connected server found for {host}:{port}")
                return []
            
            # Perform search using the remote_search method
            result = self.remote_search(node_id, query, {'k': k})
            
            if result.get('success'):
                return result.get('results', {}).get('results', [])
            else:
                self.logger.error(f"Remote search failed: {result.get('error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Search remote failed: {e}")
            return []
    
    def disconnect(self, host: str, port: int):
        """Disconnect from server by host and port (compatibility method for GUI)"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if node_id:
                self.disconnect_server(node_id)
                self.logger.info(f"Disconnected from server: {host}:{port}")
            else:
                self.logger.warning(f"No server found to disconnect: {host}:{port}")
                
        except Exception as e:
            self.logger.error(f"Disconnect failed: {e}")
    
    def get_folder_files(self, host: str, port: int) -> List[Dict[str, Any]]:
        """Get folder files from remote server (placeholder implementation)"""
        try:
            # Find node_id by host and port
            node_id = None
            for nid, server_info in self.connected_servers.items():
                if server_info['ip_address'] == host and server_info['port'] == port:
                    node_id = nid
                    break
            
            if not node_id:
                self.logger.error(f"No connected server found for {host}:{port}")
                return []
            
            # For now, return placeholder data indicating folders can be browsed
            return [
                {
                    'name': 'keyframes',
                    'path': '/shared/keyframes',
                    'type': 'directory',
                    'size': 0,
                    'modified': datetime.now().isoformat()
                },
                {
                    'name': 'exports', 
                    'path': '/shared/exports',
                    'type': 'directory',
                    'size': 0,
                    'modified': datetime.now().isoformat()
                },
                {
                    'name': 'indexes',
                    'path': '/shared/indexes', 
                    'type': 'directory',
                    'size': 0,
                    'modified': datetime.now().isoformat()
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Get folder files failed: {e}")
            return []


class RateLimiter:
    """
    🚦 Rate Limiting System
    
    Simple sliding window rate limiter for API protection.
    """
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(deque)  # client_id -> deque of timestamps
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        
        with self._lock:
            client_requests = self.requests[client_id]
            
            # Remove old requests outside window
            while client_requests and client_requests[0] < current_time - self.window_seconds:
                client_requests.popleft()
            
            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(current_time)
                return True
            
            return False
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        with self._lock:
            client_requests = self.requests[client_id]
            current_time = time.time()
            
            # Clean old requests
            while client_requests and client_requests[0] < current_time - self.window_seconds:
                client_requests.popleft()
            
            return max(0, self.max_requests - len(client_requests))


class RequestHandler:
    """
    📋 Request Handling and Validation
    
    Centralized request processing with validation, error handling, and logging.
    """
    
    def __init__(self, 
                 logger: Optional[Logger] = None,
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize request handler
        
        Args:
            logger: Logger instance
            rate_limiter: Rate limiter instance
        """
        self.logger = logger or Logger()
        self.rate_limiter = rate_limiter
        self.perf_monitor = PerformanceMonitor(self.logger)
        
        # Request validation schemas
        self.schemas = {
            "search": {
                "required": ["query"],
                "optional": ["mode", "limit", "include_temporal_context", "include_explanations", "detail_level"]
            },
            "translate": {
                "required": ["text"],
                "optional": ["target_lang", "source_lang"]
            },
            "image_search": {
                "required": ["folder_name", "image_name"],
                "optional": ["limit", "include_temporal_context", "detail_level"]
            },
            "chat": {
                "required": ["question"],
                "optional": ["context_frames"]
            },
            "metadata": {
                "required": ["folder_name", "image_name"],
                "optional": ["detail_level", "include_relationships"]
            },
            "semantic_search": {
                "required": ["semantic_query"],
                "optional": ["limit", "confidence_threshold", "detail_level"]
            }
        }
    
    def validate_request(self, api_request: APIRequest) -> APIResponse:
        """
        Validate API request
        
        Args:
            api_request: Request to validate
            
        Returns:
            APIResponse with validation result
        """
        try:
            # Rate limiting check
            if self.rate_limiter and not self.rate_limiter.is_allowed(api_request.client_id):
                return APIResponse(
                    success=False,
                    error="rate_limit_exceeded",
                    message="Too many requests. Please try again later.",
                    request_id=api_request.request_id
                )
            
            # Schema validation
            if api_request.endpoint in self.schemas:
                schema = self.schemas[api_request.endpoint]
                
                # Check required fields
                for field in schema["required"]:
                    if field not in api_request.data:
                        return APIResponse(
                            success=False,
                            error="missing_required_field",
                            message=f"Missing required field: {field}",
                            request_id=api_request.request_id
                        )
                
                # Validate field types and values
                validation_error = self._validate_field_values(api_request.data, api_request.endpoint)
                if validation_error:
                    return APIResponse(
                        success=False,
                        error="validation_error",
                        message=validation_error,
                        request_id=api_request.request_id
                    )
            
            return APIResponse(
                success=True,
                message="Request validated successfully",
                request_id=api_request.request_id
            )
            
        except Exception as e:
            self.logger.error(f"Request validation failed", error=str(e), exc_info=True)
            return APIResponse(
                success=False,
                error="validation_error",
                message="Request validation failed",
                request_id=api_request.request_id
            )
    
    def _validate_field_values(self, data: Dict[str, Any], endpoint: str) -> Optional[str]:
        """Validate specific field values"""
        if endpoint == "search":
            if "mode" in data and data["mode"] not in ["clip_only", "llm_enhanced", "hybrid"]:
                return f"Invalid search mode: {data['mode']}"
            if "limit" in data and (not isinstance(data["limit"], int) or data["limit"] <= 0):
                return "Limit must be a positive integer"
            if "detail_level" in data and data["detail_level"] not in ["minimal", "standard", "rich", "full"]:
                return f"Invalid detail level: {data['detail_level']}"
        
        elif endpoint == "translate":
            if "text" in data and not isinstance(data["text"], str):
                return "Text must be a string"
            if "text" in data and len(data["text"]) > 5000:
                return "Text too long (max 5000 characters)"
        
        elif endpoint in ["image_search", "metadata"]:
            if "detail_level" in data and data["detail_level"] not in ["minimal", "standard", "rich", "full"]:
                return f"Invalid detail level: {data['detail_level']}"
        
        elif endpoint == "semantic_search":
            if "confidence_threshold" in data and not (0 <= data["confidence_threshold"] <= 1):
                return "Confidence threshold must be between 0 and 1"
            if "detail_level" in data and data["detail_level"] not in ["minimal", "standard", "rich", "full"]:
                return f"Invalid detail level: {data['detail_level']}"
        
        return None
    
    def format_response(self, 
                       results: Any, 
                       request_id: str,
                       format_type: str = "standard",
                       detail_level: str = "standard") -> APIResponse:
        """
        Format response data with flexible detail levels
        
        Args:
            results: Result data to format
            request_id: Request identifier
            format_type: Response format type
            detail_level: Level of detail for metadata
            
        Returns:
            Formatted API response
        """
        try:
            if format_type == "search_results":
                formatted_data = self._format_search_results(results, detail_level)
            elif format_type == "translation":
                formatted_data = self._format_translation_result(results)
            elif format_type == "chat":
                formatted_data = self._format_chat_response(results)
            elif format_type == "metadata":
                formatted_data = self._format_metadata_response(results, detail_level)
            else:
                formatted_data = results
            
            return APIResponse(
                success=True,
                data=formatted_data,
                request_id=request_id
            )
            
        except Exception as e:
            self.logger.error(f"Response formatting failed", error=str(e))
            return APIResponse(
                success=False,
                error="formatting_error",
                message="Failed to format response",
                request_id=request_id
            )
    
    def _format_search_results(self, results: List[SearchResult], detail_level: str = "standard") -> Dict[str, Any]:
        """
        Format search results for API response with flexible detail levels
        
        Args:
            results: Search results to format
            detail_level: "minimal", "standard", "rich", "full"
        """
        formatted_results = {}
        
        for i, result in enumerate(results):
            key = f"rank{i+1}"
            
            # Base information (always included)
            base_info = {
                "keyframe": result.metadata.folder_name,
                "name": f"{result.metadata.image_name}.jpg",
                "frameid": result.metadata.frame_id,
                "similarity_score": round(result.similarity_score, 4)
            }
            
            # Add detail based on level
            if detail_level == "minimal":
                formatted_results[key] = base_info
                
            elif detail_level == "standard":
                # Standard level (backwards compatible)
                formatted_results[key] = {
                    **base_info,
                    "file_path": result.metadata.file_path,
                    "rank": result.rank,
                    "query_relevance": round(result.query_relevance, 4)
                }
                
                # Add explanation if available
                if result.explanation:
                    formatted_results[key]["explanation"] = result.explanation
                    
            elif detail_level == "rich":
                # Rich metadata level
                formatted_results[key] = {
                    **base_info,
                    "file_path": result.metadata.file_path,
                    "rank": result.rank,
                    "query_relevance": round(result.query_relevance, 4),
                    
                    # Semantic information
                    "scene_tags": result.metadata.scene_tags or [],
                    "detected_objects": result.metadata.detected_objects or [],
                    "confidence_score": result.metadata.confidence_score,
                    "llm_description": result.metadata.llm_description,
                    
                    # Temporal information
                    "sequence_position": result.metadata.sequence_position,
                    "total_frames": result.metadata.total_frames,
                    "neighboring_frames": result.metadata.neighboring_frames or [],
                    
                    # Explanation
                    "explanation": result.explanation
                }
                
            elif detail_level == "full":
                # Complete metadata
                formatted_results[key] = {
                    **base_info,
                    **result.metadata.to_dict(),  # Full metadata
                    "rank": result.rank,
                    "query_relevance": round(result.query_relevance, 4),
                    "explanation": result.explanation,
                    
                    # Additional computed fields
                    "has_temporal_context": len(result.temporal_context) > 0,
                    "temporal_context_count": len(result.temporal_context)
                }
            
            # Add temporal context for standard+ levels
            if detail_level in ["standard", "rich", "full"] and result.temporal_context:
                for j, context in enumerate(result.temporal_context):
                    ctx_key = f"near{i+1}.{j+1}"
                    ctx_info = {
                        "keyframe": context.metadata.folder_name,
                        "name": f"{context.metadata.image_name}.jpg",
                        "frameid": context.metadata.frame_id
                    }
                    
                    # Add rich context for rich+ levels
                    if detail_level in ["rich", "full"]:
                        ctx_info.update({
                            "scene_tags": context.metadata.scene_tags or [],
                            "detected_objects": context.metadata.detected_objects or [],
                            "sequence_position": context.metadata.sequence_position
                        })
                    
                    formatted_results[ctx_key] = ctx_info
        
        return formatted_results
    
    def _format_translation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format translation result"""
        return {
            "translated_text": result.get("translated_text", ""),
            "source_language": result.get("source_language", "auto"),
            "target_language": result.get("target_language", "en"),
            "confidence": result.get("confidence", 0.0)
        }
    
    def _format_chat_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format chat response"""
        return {
            "answer": result.get("answer", ""),
            "related_frames_count": len(result.get("related_frames", [])),
            "expanded_queries": result.get("expanded_queries", []),
            "context_used": result.get("context_used", 0)
        }
    
    def _format_metadata_response(self, metadata: KeyframeMetadata, detail_level: str = "standard") -> Dict[str, Any]:
        """
        Format metadata response with flexible detail levels
        
        Args:
            metadata: Keyframe metadata to format
            detail_level: Level of detail ("minimal", "standard", "rich", "full")
        """
        if detail_level == "minimal":
            return {
                "folder_name": metadata.folder_name,
                "image_name": metadata.image_name,
                "frame_id": metadata.frame_id
            }
        
        elif detail_level == "standard":
            return {
                "folder_name": metadata.folder_name,
                "image_name": metadata.image_name,
                "frame_id": metadata.frame_id,
                "file_path": metadata.file_path,
                "sequence_position": metadata.sequence_position,
                "total_frames": metadata.total_frames,
                "confidence_score": metadata.confidence_score
            }
        
        elif detail_level == "rich":
            return {
                "folder_name": metadata.folder_name,
                "image_name": metadata.image_name,
                "frame_id": metadata.frame_id,
                "file_path": metadata.file_path,
                
                # Temporal information
                "sequence_position": metadata.sequence_position,
                "total_frames": metadata.total_frames,
                "neighboring_frames": metadata.neighboring_frames or [],
                "scene_boundaries": metadata.scene_boundaries or [],
                
                # Semantic information
                "scene_tags": metadata.scene_tags or [],
                "detected_objects": metadata.detected_objects or [],
                "llm_description": metadata.llm_description,
                "confidence_score": metadata.confidence_score,
                
                # Basic relationships
                "similar_frames_count": len(metadata.similar_frames or []),
                "transition_frames_count": len(metadata.transition_frames or [])
            }
        
        elif detail_level == "full":
            # Complete metadata with all relationships
            return metadata.to_dict()
        
        else:
            return metadata.to_dict()
    
    def handle_error(self, error: Exception, request_id: str) -> APIResponse:
        """
        Handle and format errors
        
        Args:
            error: Exception that occurred
            request_id: Request identifier
            
        Returns:
            Error API response
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error
        self.logger.error(f"Request error", 
                         error_type=error_type,
                         error_message=error_message,
                         request_id=request_id,
                         exc_info=True)
        
        # Map specific errors to user-friendly messages
        if isinstance(error, FileNotFoundError):
            message = "Requested resource not found"
        elif isinstance(error, ValueError):
            message = f"Invalid input: {error_message}"
        elif isinstance(error, RuntimeError):
            message = "System temporarily unavailable"
        else:
            message = "An internal error occurred"
        
        return APIResponse(
            success=False,
            error=error_type.lower(),
            message=message,
            request_id=request_id
        )


class TranslationService:
    """
    🌍 Translation Service Integration
    
    Google Translate integration with caching and error handling.
    """
    
    def __init__(self, 
                 logger: Optional[Logger] = None,
                 cache: Optional[CacheManager] = None):
        """
        Initialize translation service
        
        Args:
            logger: Logger instance
            cache: Cache manager for translations
        """
        self.logger = logger or Logger()
        self.cache = cache or CacheManager()
        self.perf_monitor = PerformanceMonitor(self.logger)
        
        # Initialize translator
        if HAS_TRANSLATION:
            self.translator = Translator()
            self.nlp = None
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.warning("spaCy English model not found")
        else:
            self.translator = None
            self.nlp = None
            self.logger.warning("Translation dependencies not available")
    
    def translate_text(self, 
                      text: str, 
                      target_lang: str = "en",
                      source_lang: str = "auto") -> Dict[str, Any]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detect if 'auto')
            
        Returns:
            Translation result dictionary
        """
        if not self.translator:
            return {
                "translated_text": text,
                "source_language": "unknown",
                "target_language": target_lang,
                "confidence": 0.0,
                "error": "Translation service not available"
            }
        
        # Check cache first
        cache_key = f"translate:{source_lang}:{target_lang}:{hash(text)}"
        cached_result = self.cache.get(cache_key, ttl=86400)  # 24 hours
        if cached_result:
            return cached_result
        
        try:
            with self.perf_monitor.timer("translate_text"):
                # Perform translation
                if source_lang == "auto":
                    result = self.translator.translate(text, dest=target_lang)
                    detected_lang = result.src
                else:
                    result = self.translator.translate(text, src=source_lang, dest=target_lang)
                    detected_lang = source_lang
                
                translation_result = {
                    "translated_text": result.text,
                    "source_language": detected_lang,
                    "target_language": target_lang,
                    "confidence": getattr(result, 'confidence', 0.0)
                }
                
                # Cache result
                self.cache.set(cache_key, translation_result, persist=True)
                
                self.logger.debug(f"Text translated",
                                source_lang=detected_lang,
                                target_lang=target_lang,
                                text_length=len(text))
                
                return translation_result
                
        except Exception as e:
            self.logger.error(f"Translation failed", error=str(e))
            return {
                "translated_text": text,
                "source_language": "unknown", 
                "target_language": target_lang,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        if not self.translator:
            return "unknown"
        
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except Exception as e:
            self.logger.error(f"Language detection failed", error=str(e))
            return "unknown"
    
    def extract_text_features(self, text: str) -> Dict[str, List[str]]:
        """
        Extract linguistic features from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        if not self.nlp:
            # Fallback to simple analysis
            return {
                "tokens": text.split(),
                "numbers": [],
                "letters": [],
                "punctuation": []
            }
        
        try:
            doc = self.nlp(text)
            
            return {
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "numbers": [token.text for token in doc if token.pos_ == 'NUM'],
                "letters": [token.text for token in doc if len(token.text) == 1 and token.text.isalpha()],
                "punctuation": [token.text for token in doc if token.pos_ == 'PUNCT']
            }
            
        except Exception as e:
            self.logger.error(f"Text feature extraction failed", error=str(e))
            return {"tokens": text.split(), "numbers": [], "letters": [], "punctuation": []}


class SocketIOServer:
    """
    🔌 Real-time SocketIO Communication Server
    
    Enhanced SocketIO server with advanced features and better error handling.
    """
    
    def __init__(self, 
                 retrieval_system: EnhancedRetrievalSystem,
                 config: Optional[Config] = None,
                 logger: Optional[Logger] = None):
        """
        Initialize SocketIO server
        
        Args:
            retrieval_system: Main retrieval system instance
            config: System configuration
            logger: Logger instance
        """
        if not HAS_SOCKETIO:
            raise ImportError("SocketIO dependencies not available. Install python-socketio and eventlet.")
        
        self.system = retrieval_system
        self.config = config or Config()
        self.logger = logger or Logger()
        
        # Initialize components
        self.translation_service = TranslationService(self.logger, self.system.cache)
        self.request_handler = RequestHandler(
            self.logger,
            RateLimiter(
                max_requests=self.config.get("api.rate_limit", 100),
                window_minutes=1
            )
        )
        
        # Initialize SocketIO server
        self.sio = socketio.Server(
            cors_allowed_origins="*" if self.config.get("api.cors_enabled", True) else None,
            logger=False,  # Use our custom logger
            engineio_logger=False
        )
        
        # Create WSGI app
        self.app = socketio.WSGIApp(self.sio)
        
        # Connection tracking
        self.connected_clients = {}
        self.client_sessions = defaultdict(dict)
        
        # Setup event handlers
        self._setup_events()
        
        self.logger.info("SocketIO server initialized")
    
    def _setup_events(self) -> None:
        """Setup SocketIO event handlers"""
        
        @self.sio.event
        def connect(sid, environ):
            """Handle client connection"""
            client_info = {
                "sid": sid,
                "connected_at": time.time(),
                "remote_addr": environ.get('REMOTE_ADDR', 'unknown'),
                "user_agent": environ.get('HTTP_USER_AGENT', 'unknown')
            }
            
            self.connected_clients[sid] = client_info
            self.logger.info(f"Client connected", 
                           sid=sid, 
                           remote_addr=client_info["remote_addr"])
            
            # Send welcome message
            self.sio.emit('connected', {
                'message': 'Connected to Enhanced Retrieval System',
                'server_version': '2.0',
                'capabilities': ['search', 'translate', 'chat', 'image_search']
            }, room=sid)
        
        @self.sio.event
        def disconnect(sid):
            """Handle client disconnection"""
            if sid in self.connected_clients:
                client_info = self.connected_clients.pop(sid)
                duration = time.time() - client_info["connected_at"]
                self.logger.info(f"Client disconnected", 
                               sid=sid, 
                               duration=duration)
            
            # Cleanup client session
            if sid in self.client_sessions:
                del self.client_sessions[sid]
        
        @self.sio.event
        def translated_text(sid, data):
            """Handle translated text search request"""
            self._handle_search_request(sid, data, "translated_text")
        
        @self.sio.event
        def search(sid, data):
            """Handle direct search request"""
            self._handle_search_request(sid, data, "search")
        
        @self.sio.event
        def translate(sid, data):
            """Handle translation request"""
            self._handle_translation_request(sid, data)
        
        @self.sio.event
        def image_search(sid, data):
            """Handle image-based search request"""
            self._handle_image_search_request(sid, data)
        
        @self.sio.event
        def chat(sid, data):
            """Handle chat/conversation request"""
            self._handle_chat_request(sid, data)
        
        @self.sio.event
        def metadata_query(sid, data):
            """Handle metadata query request"""
            self._handle_metadata_request(sid, data)
        
        @self.sio.event
        def semantic_search(sid, data):
            """Handle semantic search request"""
            self._handle_semantic_search_request(sid, data)
        
        @self.sio.event
        def system_stats(sid, data):
            """Handle system statistics request"""
            self._handle_system_stats_request(sid, data)
        
        @self.sio.event
        def get_surrounding_frames(sid, data):
            """Handle surrounding frames request"""
            self._handle_surrounding_frames_request(sid, data)
        
        @self.sio.event
        def get_frame_context(sid, data):
            """Handle frame context request"""
            self._handle_frame_context_request(sid, data)
        
        @self.sio.event
        def export_frames(sid, data):
            """Handle export frames request"""
            self._handle_export_frames_request(sid, data)
    
    def _handle_search_request(self, sid: str, data: Dict[str, Any], event_type: str) -> None:
        """Handle search requests"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            # Create API request
            api_request = APIRequest(
                endpoint="search",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            # Validate request
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Handle translation if needed
            query_text = data.get('text', data.get('query', ''))
            if event_type == "translated_text":
                translation_result = self.translation_service.translate_text(query_text)
                query_text = translation_result["translated_text"]
                
                # Emit translation result
                self.sio.emit('translation_result', translation_result, room=sid)
            
            # Perform search
            search_options = SearchOptions(
                mode=data.get('mode', 'hybrid'),
                limit=data.get('limit', 50),
                include_temporal_context=data.get('include_temporal_context', True),
                include_explanations=data.get('include_explanations', False)
            )
            
            search_results = self.system.search(query_text, search_options)
            
            # Get detail level from request
            detail_level = data.get('detail_level', 'standard')
            
            # Format and send results
            response = self.request_handler.format_response(
                search_results, 
                request_id, 
                "search_results",
                detail_level
            )
            
            if response.success:
                self.sio.emit('result', response.data, room=sid)
                
                # Update client session
                self.client_sessions[sid]['last_search'] = {
                    'query': query_text,
                    'results_count': len(search_results),
                    'timestamp': time.time()
                }
            else:
                self.sio.emit('error', response.to_dict(), room=sid)
                
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _handle_translation_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle translation requests"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            api_request = APIRequest(
                endpoint="translate",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Perform translation
            result = self.translation_service.translate_text(
                data['text'],
                data.get('target_lang', 'en'),
                data.get('source_lang', 'auto')
            )
            
            # Format response
            response = self.request_handler.format_response(result, request_id, "translation")
            self.sio.emit('translation_result', response.data, room=sid)
            
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _handle_image_search_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle image-based search requests"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            api_request = APIRequest(
                endpoint="image_search",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Perform image search
            search_options = SearchOptions(
                limit=data.get('limit', 50),
                include_temporal_context=data.get('include_temporal_context', True)
            )
            
            results = self.system.search_by_image(
                data['folder_name'],
                data['image_name'],
                search_options
            )
            
            # Format and send results
            response = self.request_handler.format_response(results, request_id, "search_results")
            self.sio.emit('image_search_result', response.data, room=sid)
            
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _handle_chat_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle chat/conversation requests"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            api_request = APIRequest(
                endpoint="chat",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Perform chat search
            result = self.system.chat_search(
                data['question'],
                data.get('context_frames')
            )
            
            # Format and send response
            response = self.request_handler.format_response(result, request_id, "chat")
            self.sio.emit('chat_response', response.data, room=sid)
            
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _handle_system_stats_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle system statistics requests"""
        try:
            stats = self.system.get_system_stats()
            
            # Add server-specific stats
            server_stats = {
                "connected_clients": len(self.connected_clients),
                "server_uptime": time.time() - getattr(self, 'start_time', time.time()),
                "total_requests": sum(len(self.request_handler.perf_monitor.operation_stats.get(op, [])) 
                                    for op in ['search_query', 'translate_text'])
            }
            
            stats.update({"server": server_stats})
            
            self.sio.emit('system_stats', stats, room=sid)
            
        except Exception as e:
            self.logger.error(f"System stats request failed", error=str(e))
            self.sio.emit('error', {'error': 'stats_unavailable', 'message': str(e)}, room=sid)
    
    def _handle_metadata_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle detailed metadata requests"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            api_request = APIRequest(
                endpoint="metadata",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Get metadata from system
            metadata = self.system.metadata_manager.get_metadata(
                data['folder_name'],
                data['image_name']
            )
            
            if not metadata:
                self.sio.emit('error', {
                    'error': 'metadata_not_found',
                    'message': f"Metadata not found for {data['folder_name']}/{data['image_name']}"
                }, room=sid)
                return
            
            # Format response with requested detail level
            detail_level = data.get('detail_level', 'rich')
            response = self.request_handler.format_response(
                metadata, 
                request_id, 
                "metadata",
                detail_level
            )
            
            # Add relationships if requested
            if data.get('include_relationships', False):
                relationships = {
                    "temporal_neighbors": self.system.metadata_manager.get_temporal_neighbors(
                        data['folder_name'], 
                        data['image_name']
                    ),
                    "similar_frames": self.system.metadata_manager.get_similar_frames(
                        data['folder_name'], 
                        data['image_name']
                    )
                }
                response.data['relationships'] = relationships
            
            self.sio.emit('metadata_result', response.data, room=sid)
            
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _handle_semantic_search_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle semantic search requests using scene tags and detected objects"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            api_request = APIRequest(
                endpoint="semantic_search",
                data=data,
                client_id=sid,
                timestamp=time.time(),
                request_id=request_id
            )
            
            validation_result = self.request_handler.validate_request(api_request)
            if not validation_result.success:
                self.sio.emit('error', validation_result.to_dict(), room=sid)
                return
            
            # Extract semantic query
            semantic_query = data['semantic_query']
            limit = data.get('limit', 50)
            confidence_threshold = data.get('confidence_threshold', 0.5)
            detail_level = data.get('detail_level', 'rich')
            
            # Perform semantic search using metadata
            semantic_results = self._perform_semantic_search(
                semantic_query, limit, confidence_threshold
            )
            
            # Format response
            response = self.request_handler.format_response(
                semantic_results, 
                request_id, 
                "search_results",
                detail_level
            )
            
            self.sio.emit('semantic_search_result', response.data, room=sid)
            
        except Exception as e:
            error_response = self.request_handler.handle_error(e, request_id)
            self.sio.emit('error', error_response.to_dict(), room=sid)
    
    def _perform_semantic_search(self, 
                                query: str, 
                                limit: int, 
                                confidence_threshold: float) -> List[SearchResult]:
        """
        Perform semantic search based on scene tags and detected objects
        
        Args:
            query: Semantic query (e.g., "cars", "people talking", "outdoor scenes")
            limit: Maximum results to return
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of matching search results
        """
        query_terms = set(query.lower().split())
        results = []
        
        # Search through all metadata
        for folder_name, folder_metadata in self.system.metadata_manager.metadata_db.items():
            for image_name, metadata in folder_metadata.items():
                score = 0.0
                
                # Check scene tags
                if metadata.scene_tags:
                    tag_terms = set(tag.lower() for tag in metadata.scene_tags)
                    tag_matches = len(query_terms.intersection(tag_terms))
                    score += tag_matches * 0.4
                
                # Check detected objects
                if metadata.detected_objects:
                    object_terms = set(obj.lower() for obj in metadata.detected_objects)
                    object_matches = len(query_terms.intersection(object_terms))
                    score += object_matches * 0.5
                
                # Check LLM description
                if metadata.llm_description:
                    desc_terms = set(metadata.llm_description.lower().split())
                    desc_matches = len(query_terms.intersection(desc_terms))
                    score += desc_matches * 0.3
                
                # Check folder name (scene type)
                folder_terms = set(folder_name.lower().split('_'))
                folder_matches = len(query_terms.intersection(folder_terms))
                score += folder_matches * 0.2
                
                # Add to results if above threshold
                if score >= confidence_threshold:
                    search_result = SearchResult(
                        metadata=metadata,
                        similarity_score=score,
                        ranking_score=score,
                        rank=0,  # Will be set after sorting
                        query_relevance=score
                    )
                    results.append(search_result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: getattr(x, 'ranking_score', x.similarity_score), reverse=True)
        results = results[:limit]
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _handle_surrounding_frames_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle surrounding frames request"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            # Validate required fields
            required_fields = ['folder_name', 'image_name']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                self.sio.emit('error', {
                    'error': 'missing_fields',
                    'message': f"Missing required fields: {', '.join(missing_fields)}",
                    'request_id': request_id
                }, room=sid)
                return
            
            folder_name = data['folder_name']
            image_name = data['image_name']
            window = data.get('window', 10)
            
            # Get surrounding frames
            surrounding_frames = self.system.get_surrounding_frames(
                folder_name, image_name, window
            )
            
            # Convert to serializable format
            frames_data = []
            for frame in surrounding_frames:
                frame_dict = {
                    'folder_name': frame.folder_name,
                    'image_name': frame.image_name,
                    'frame_id': frame.frame_id,
                    'file_path': frame.file_path,
                    'sequence_position': frame.sequence_position,
                    'scene_tags': frame.scene_tags,
                    'detected_objects': frame.detected_objects
                }
                frames_data.append(frame_dict)
            
            response = {
                'success': True,
                'request_id': request_id,
                'original_frame': {
                    'folder_name': folder_name,
                    'image_name': image_name
                },
                'surrounding_frames': frames_data,
                'total_frames': len(frames_data),
                'window_size': window
            }
            
            self.sio.emit('surrounding_frames_result', response, room=sid)
            
        except Exception as e:
            self.logger.error(f"Surrounding frames request failed", error=str(e))
            self.sio.emit('error', {
                'error': 'surrounding_frames_failed',
                'message': str(e),
                'request_id': request_id
            }, room=sid)
    
    def _handle_frame_context_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle frame context request with optional auto-export"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            # Validate required fields
            required_fields = ['folder_name', 'image_name']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                self.sio.emit('error', {
                    'error': 'missing_fields',
                    'message': f"Missing required fields: {', '.join(missing_fields)}",
                    'request_id': request_id
                }, room=sid)
                return
            
            folder_name = data['folder_name']
            image_name = data['image_name']
            window = data.get('window', 10)
            auto_export = data.get('auto_export', False)
            export_path = data.get('export_path', None)
            
            # Get frame context
            context_data = self.system.get_frame_context_with_export(
                folder_name, image_name, window, auto_export, export_path
            )
            
            # Convert frames to serializable format
            def frame_to_dict(frame):
                return {
                    'folder_name': frame.folder_name,
                    'image_name': frame.image_name,
                    'frame_id': frame.frame_id,
                    'file_path': frame.file_path,
                    'sequence_position': frame.sequence_position,
                    'scene_tags': frame.scene_tags,
                    'detected_objects': frame.detected_objects
                }
            
            response = {
                'success': True,
                'request_id': request_id,
                'original_frame': frame_to_dict(context_data['original_frame']),
                'surrounding_frames': [frame_to_dict(f) for f in context_data['surrounding_frames']],
                'total_frames': context_data['total_frames'],
                'video_name': context_data['video_name'],
                'window_size': context_data['window_size'],
                'exported': context_data['exported']
            }
            
            if context_data.get('export_path'):
                response['export_path'] = context_data['export_path']
            
            self.sio.emit('frame_context_result', response, room=sid)
            
        except Exception as e:
            self.logger.error(f"Frame context request failed", error=str(e))
            self.sio.emit('error', {
                'error': 'frame_context_failed',
                'message': str(e),
                'request_id': request_id
            }, room=sid)
    
    def _handle_export_frames_request(self, sid: str, data: Dict[str, Any]) -> None:
        """Handle export frames request"""
        request_id = f"{sid}_{time.time()}"
        
        try:
            # Validate required fields
            if 'frames' not in data or not data['frames']:
                self.sio.emit('error', {
                    'error': 'missing_frames',
                    'message': "Missing or empty 'frames' field",
                    'request_id': request_id
                }, room=sid)
                return
            
            frames_data = data['frames']
            output_path = data.get('output_path', None)
            include_metadata = data.get('include_metadata', True)
            
            # Convert frame data to KeyframeMetadata objects
            selected_frames = []
            for frame_data in frames_data:
                try:
                    frame = KeyframeMetadata(
                        folder_name=frame_data['folder_name'],
                        image_name=frame_data['image_name'],
                        frame_id=frame_data['frame_id'],
                        file_path=frame_data['file_path'],
                        sequence_position=frame_data.get('sequence_position', 0),
                        scene_tags=frame_data.get('scene_tags', []),
                        detected_objects=frame_data.get('detected_objects', [])
                    )
                    selected_frames.append(frame)
                except Exception as e:
                    self.sio.emit('error', {
                        'error': 'invalid_frame_data',
                        'message': f"Invalid frame data: {str(e)}",
                        'request_id': request_id
                    }, room=sid)
                    return
            
            # Export to CSV
            export_path = self.system.export_selected_frames_to_csv(
                selected_frames, output_path, include_metadata
            )
            
            response = {
                'success': True,
                'request_id': request_id,
                'export_path': export_path,
                'frames_exported': len(selected_frames),
                'include_metadata': include_metadata
            }
            
            self.sio.emit('export_frames_result', response, room=sid)
            
        except Exception as e:
            self.logger.error(f"Export frames request failed", error=str(e))
            self.sio.emit('error', {
                'error': 'export_frames_failed',
                'message': str(e),
                'request_id': request_id
            }, room=sid)
    
    def run(self, host: str = "localhost", port: int = 5000, **kwargs) -> None:
        """
        Run the SocketIO server
        
        Args:
            host: Server host
            port: Server port
            **kwargs: Additional server arguments
        """
        self.start_time = time.time()
        
        try:
            self.logger.info(f"Starting SocketIO server", host=host, port=port)
            
            # Start server
            eventlet.wsgi.server(
                eventlet.listen((host, port)), 
                self.app,
                log=None,  # Disable eventlet logging
                **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Server startup failed", error=str(e))
            raise
    
    def broadcast_message(self, event: str, data: Any, room: Optional[str] = None) -> None:
        """Broadcast message to clients"""
        try:
            self.sio.emit(event, data, room=room)
        except Exception as e:
            self.logger.error(f"Broadcast failed", event=event, error=str(e))


class RESTAPIServer:
    """
    🌐 HTTP REST API Server
    
    Modern REST API interface for HTTP-based integration.
    """
    
    def __init__(self, 
                 retrieval_system: EnhancedRetrievalSystem,
                 config: Optional[Config] = None,
                 logger: Optional[Logger] = None):
        """
        Initialize REST API server
        
        Args:
            retrieval_system: Main retrieval system instance
            config: System configuration  
            logger: Logger instance
        """
        if not HAS_FLASK:
            raise ImportError("Flask dependencies not available. Install Flask and flask-cors.")
        
        self.system = retrieval_system
        self.config = config or Config()
        self.logger = logger or Logger()
        
        # Initialize components
        self.translation_service = TranslationService(self.logger, self.system.cache)
        self.request_handler = RequestHandler(
            self.logger,
            RateLimiter(
                max_requests=self.config.get("api.rate_limit", 100),
                window_minutes=1
            )
        )
        
        # Initialize Flask app
        self.app = Flask(__name__)
        
        # Enable CORS if configured
        if self.config.get("api.cors_enabled", True):
            CORS(self.app)
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("REST API server initialized")
    
    def _setup_routes(self) -> None:
        """Setup REST API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "version": "2.0",
                "system_ready": self.system.status.is_ready,
                "timestamp": time.time()
            })
        
        @self.app.route('/api/search', methods=['POST'])
        def search():
            """Enhanced search endpoint with metadata support"""
            try:
                data = request.get_json()
                
                # Create API request
                api_request = APIRequest(
                    endpoint="search",
                    data=data,
                    client_id=request.remote_addr,
                    timestamp=time.time(),
                    request_id=f"rest_{time.time()}"
                )
                
                # Validate request
                validation_result = self.request_handler.validate_request(api_request)
                if not validation_result.success:
                    return jsonify(validation_result.to_dict()), 400
                
                # Perform search
                search_options = SearchOptions(
                    mode=data.get('mode', 'hybrid'),
                    limit=data.get('limit', 50),
                    include_temporal_context=data.get('include_temporal_context', True),
                    include_explanations=data.get('include_explanations', False)
                )
                
                results = self.system.search(data['query'], search_options)
                
                # Format response with detail level
                detail_level = data.get('detail_level', 'standard')
                response = self.request_handler.format_response(
                    results, 
                    api_request.request_id, 
                    "search_results",
                    detail_level
                )
                
                return jsonify(response.to_dict())
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "rest_search")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/metadata/<folder_name>/<image_name>', methods=['GET'])
        def get_metadata(folder_name, image_name):
            """Get detailed metadata for a specific keyframe"""
            try:
                detail_level = request.args.get('detail_level', 'rich')
                include_relationships = request.args.get('include_relationships', 'false').lower() == 'true'
                
                # Get metadata
                metadata = self.system.metadata_manager.get_metadata(folder_name, image_name)
                if not metadata:
                    return jsonify({
                        "success": False,
                        "error": "metadata_not_found",
                        "message": f"Metadata not found for {folder_name}/{image_name}"
                    }), 404
                
                # Format response
                response = self.request_handler.format_response(
                    metadata,
                    "rest_metadata",
                    "metadata",
                    detail_level
                )
                
                # Add relationships if requested
                if include_relationships:
                    relationships = {
                        "temporal_neighbors": [
                            meta.to_dict() for meta in 
                            self.system.metadata_manager.get_temporal_neighbors(folder_name, image_name)
                        ],
                        "similar_frames": [
                            meta.to_dict() for meta in
                            self.system.metadata_manager.get_similar_frames(folder_name, image_name)
                        ]
                    }
                    response.data['relationships'] = relationships
                
                return jsonify(response.to_dict())
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "rest_metadata")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/semantic_search', methods=['POST'])
        def semantic_search():
            """Semantic search using scene tags and detected objects"""
            try:
                data = request.get_json()
                
                # Validate required fields
                if 'semantic_query' not in data:
                    return jsonify({
                        "success": False,
                        "error": "missing_field",
                        "message": "semantic_query field is required"
                    }), 400
                
                # Extract parameters
                semantic_query = data['semantic_query']
                limit = data.get('limit', 50)
                confidence_threshold = data.get('confidence_threshold', 0.5)
                detail_level = data.get('detail_level', 'rich')
                
                # Perform semantic search using SocketIO server's method
                socketio_server = SocketIOServer(self.system, self.config, self.logger)
                results = socketio_server._perform_semantic_search(
                    semantic_query, limit, confidence_threshold
                )
                
                # Format response
                response = self.request_handler.format_response(
                    results,
                    "rest_semantic_search",
                    "search_results",
                    detail_level
                )
                
                return jsonify(response.to_dict())
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "rest_semantic_search")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/translate', methods=['POST'])
        def translate():
            """Translation endpoint"""
            try:
                data = request.get_json()
                
                result = self.translation_service.translate_text(
                    data['text'],
                    data.get('target_lang', 'en'),
                    data.get('source_lang', 'auto')
                )
                
                response = self.request_handler.format_response(result, "rest_translate", "translation")
                return jsonify(response.to_dict())
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "rest_translate")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def system_stats():
            """System statistics endpoint"""
            try:
                stats = self.system.get_system_stats()
                return jsonify(stats)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/export', methods=['POST'])
        def export_data():
            """Data export endpoint"""
            try:
                data = request.get_json()
                
                # This would require implementing result storage/retrieval
                # For now, return placeholder
                return jsonify({
                    "message": "Export functionality requires session management",
                    "status": "not_implemented"
                }), 501
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/frames/surrounding', methods=['POST'])
        def get_surrounding_frames():
            """Get surrounding frames (10 before and after) for a selected frame"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['folder_name', 'image_name']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": f"Missing required fields: {', '.join(missing_fields)}"
                    }), 400
                
                folder_name = data['folder_name']
                image_name = data['image_name']
                window = data.get('window', 10)  # Default 10 frames before/after
                
                # Get surrounding frames
                surrounding_frames = self.system.get_surrounding_frames(
                    folder_name, image_name, window
                )
                
                # Convert to serializable format
                frames_data = []
                for frame in surrounding_frames:
                    frame_dict = {
                        'folder_name': frame.folder_name,
                        'image_name': frame.image_name,
                        'frame_id': frame.frame_id,
                        'file_path': frame.file_path,
                        'sequence_position': frame.sequence_position,
                        'scene_tags': frame.scene_tags,
                        'detected_objects': frame.detected_objects
                    }
                    frames_data.append(frame_dict)
                
                response = {
                    'success': True,
                    'original_frame': {
                        'folder_name': folder_name,
                        'image_name': image_name
                    },
                    'surrounding_frames': frames_data,
                    'total_frames': len(frames_data),
                    'window_size': window
                }
                
                return jsonify(response)
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "get_surrounding_frames")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/frames/context', methods=['POST'])
        def get_frame_context():
            """Get frame context with surrounding frames and optional auto-export"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['folder_name', 'image_name']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": f"Missing required fields: {', '.join(missing_fields)}"
                    }), 400
                
                folder_name = data['folder_name']
                image_name = data['image_name']
                window = data.get('window', 10)
                auto_export = data.get('auto_export', False)
                export_path = data.get('export_path', None)
                
                # Get frame context
                context_data = self.system.get_frame_context_with_export(
                    folder_name, image_name, window, auto_export, export_path
                )
                
                # Convert frames to serializable format
                def frame_to_dict(frame):
                    return {
                        'folder_name': frame.folder_name,
                        'image_name': frame.image_name,
                        'frame_id': frame.frame_id,
                        'file_path': frame.file_path,
                        'sequence_position': frame.sequence_position,
                        'scene_tags': frame.scene_tags,
                        'detected_objects': frame.detected_objects
                    }
                
                response = {
                    'success': True,
                    'original_frame': frame_to_dict(context_data['original_frame']),
                    'surrounding_frames': [frame_to_dict(f) for f in context_data['surrounding_frames']],
                    'total_frames': context_data['total_frames'],
                    'video_name': context_data['video_name'],
                    'window_size': context_data['window_size'],
                    'exported': context_data['exported']
                }
                
                if context_data.get('export_path'):
                    response['export_path'] = context_data['export_path']
                
                return jsonify(response)
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "get_frame_context")
                return jsonify(error_response.to_dict()), 500
        
        @self.app.route('/api/frames/export', methods=['POST'])
        def export_selected_frames():
            """Export selected frames to CSV"""
            try:
                data = request.get_json()
                
                # Validate required fields
                if 'frames' not in data or not data['frames']:
                    return jsonify({
                        "error": "Missing or empty 'frames' field"
                    }), 400
                
                frames_data = data['frames']
                output_path = data.get('output_path', None)
                include_metadata = data.get('include_metadata', True)
                
                # Convert frame data to KeyframeMetadata objects
                selected_frames = []
                for frame_data in frames_data:
                    try:
                        frame = KeyframeMetadata(
                            folder_name=frame_data['folder_name'],
                            image_name=frame_data['image_name'],
                            frame_id=frame_data['frame_id'],
                            file_path=frame_data['file_path'],
                            sequence_position=frame_data.get('sequence_position', 0),
                            scene_tags=frame_data.get('scene_tags', []),
                            detected_objects=frame_data.get('detected_objects', [])
                        )
                        selected_frames.append(frame)
                    except Exception as e:
                        return jsonify({
                            "error": f"Invalid frame data: {str(e)}"
                        }), 400
                
                # Export to CSV
                export_path = self.system.export_selected_frames_to_csv(
                    selected_frames, output_path, include_metadata
                )
                
                response = {
                    'success': True,
                    'export_path': export_path,
                    'frames_exported': len(selected_frames),
                    'include_metadata': include_metadata
                }
                
                return jsonify(response)
                
            except Exception as e:
                error_response = self.request_handler.handle_error(e, "export_selected_frames")
                return jsonify(error_response.to_dict()), 500
    
    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False) -> None:
        """
        Run the REST API server
        
        Args:
            host: Server host
            port: Server port  
            debug: Enable debug mode
        """
        try:
            self.logger.info(f"Starting REST API server", host=host, port=port)
            self.app.run(host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"REST API server startup failed", error=str(e))
            raise


# Module-level convenience functions
def create_socketio_server(retrieval_system: EnhancedRetrievalSystem, **kwargs) -> SocketIOServer:
    """Create SocketIO server instance"""
    return SocketIOServer(retrieval_system, **kwargs)

def create_rest_server(retrieval_system: EnhancedRetrievalSystem, **kwargs) -> RESTAPIServer:
    """Create REST API server instance"""
    return RESTAPIServer(retrieval_system, **kwargs)

def create_translation_service(**kwargs) -> TranslationService:
    """Create translation service instance"""
    return TranslationService(**kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rest':
        # Start REST API server
        print("🌐 Starting Enhanced Retrieval System REST API Server")
        print("=" * 60)
        
        try:
            from utils import Config, Logger
            config = Config()
            logger = Logger()
            
            server = RESTAPIServer(config, logger)
            print(f"🚀 Starting server on port 5000...")
            server.run(host="0.0.0.0", port=5000)
            
        except ImportError as e:
            print(f"Error importing dependencies: {e}")
            print("Starting minimal server...")
            
            if HAS_FLASK:
                app = Flask(__name__)
                CORS(app)
                
                @app.route('/health')
                def health():
                    return jsonify({"status": "healthy", "service": "enhanced-retrieval"})
                
                @app.route('/')
                def home():
                    return jsonify({
                        "service": "Enhanced Retrieval System",
                        "status": "running",
                        "version": "3.0-docker"
                    })
                
                app.run(host="0.0.0.0", port=5000, debug=False)
            else:
                print("Flask not available")
                
    elif len(sys.argv) > 1 and sys.argv[1] == '--socketio':
        # Start SocketIO server  
        print("🌐 Starting Enhanced Retrieval System SocketIO Server")
        print("=" * 60)
        
        try:
            from utils import Config, Logger
            config = Config()
            logger = Logger()
            
            server = SocketIOServer(config, logger)
            server.run()
            
        except ImportError as e:
            print(f"Error: {e}")
            
    else:
        # Example usage and testing
        print("🌐 Enhanced Retrieval System - API & Communication Layer")
        print("=" * 60)
        
        # Test components without full system
    print("\n🔧 Testing API Components...")
    
    # Test rate limiter
    rate_limiter = RateLimiter(max_requests=5, window_minutes=1)
    for i in range(7):
        allowed = rate_limiter.is_allowed("test_client")
        print(f"Request {i+1}: {'✅ Allowed' if allowed else '❌ Rate limited'}")
    
    # Test request handler
    logger = Logger()
    request_handler = RequestHandler(logger, rate_limiter)
    
    test_request = APIRequest(
        endpoint="search",
        data={"query": "test search"},
        client_id="test_client",
        timestamp=time.time(),
        request_id="test_123"
    )
    
    validation_result = request_handler.validate_request(test_request)
    print(f"Request validation: {'✅ Valid' if validation_result.success else '❌ Invalid'}")
    
    # Test translation service
    if HAS_TRANSLATION:
        translation_service = TranslationService(logger)
        result = translation_service.translate_text("Hello world", "es")
        print(f"Translation test: {result.get('translated_text', 'Failed')}")
    else:
        print("❌ Translation service not available (missing dependencies)")
    
    print(f"\n📊 Available features:")
    print(f"- SocketIO Server: {'✅' if HAS_SOCKETIO else '❌'}")
    print(f"- REST API Server: {'✅' if HAS_FLASK else '❌'}")
    print(f"- Translation Service: {'✅' if HAS_TRANSLATION else '❌'}")
    
    print("\n✅ API module tested successfully!")
    print("\nTo run servers:")
    print("1. SocketIO: python api.py --socketio")
    print("2. REST API: python api.py --rest")
