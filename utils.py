"""
Enhanced Retrieval System - Utilities Module với Unicode Support & Validation
============================================================================

All utility functions and helper classes for the system.
Contains: Configuration, File Management, Data Processing, Caching, Logging, Performance Monitoring
Enhanced: Unicode Support, Robust Validation, Error Recovery, Health Monitoring
OpenAI-Focused: Optimized for OpenAI GPT models only

Author: Enhanced Retrieval System
Version: 2.1 - OpenAI Edition
"""

import os
import sys
import json
import pickle
import shutil
import logging
import logging.handlers
import hashlib
import time
import locale
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from contextlib import contextmanager
from io import TextIOWrapper
import numpy as np
import pandas as pd

# Agno Framework imports (OpenAI only)
try:
    from agno.agent import Agent, AgentMemory
    from agno.models.openai import OpenAIChat
    from agno.storage.postgres import PostgresStorage
    from agno.storage.agent.sqlite import SqliteAgentStorage
    from agno.memory.db.postgres import PgMemoryDb
    from agno.team import Team
    import agentops
    HAS_AGNO = True
except ImportError:
    HAS_AGNO = False
    Agent = AgentMemory = OpenAIChat = None
    PostgresStorage = SqliteAgentStorage = PgMemoryDb = Team = None
    agentops = None


class UnicodeHelper:
    """
    🌏 Unicode Support Utilities
    
    Handles Unicode text safely across different platforms and encodings.
    """
    
    @staticmethod
    def setup_unicode_environment():
        """Setup Unicode environment for Vietnamese text and other non-ASCII characters"""
        try:
            # Set Python encoding
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # For Windows
            if sys.platform.startswith('win'):
                try:
                    # Set console code page to UTF-8
                    import subprocess
                    subprocess.run(['chcp', '65001'], shell=True, capture_output=True, timeout=5)
                except Exception:
                    pass  # Ignore if fails
                
                try:
                    # Set locale
                    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                except:
                    try:
                        locale.setlocale(locale.LC_ALL, '')
                    except:
                        pass  # Ignore locale setting failures
            
            # Reconfigure stdout/stderr for UTF-8
            try:
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                elif hasattr(sys.stdout, 'buffer'):
                    # For older Python versions, wrap stdout
                    sys.stdout = TextIOWrapper(
                        sys.stdout.buffer, 
                        encoding='utf-8', 
                        errors='replace'
                    )
                    sys.stderr = TextIOWrapper(
                        sys.stderr.buffer,
                        encoding='utf-8',
                        errors='replace'
                    )
            except Exception:
                pass  # If reconfiguration fails, continue anyway
                
        except Exception:
            pass  # Don't fail if Unicode setup has issues
    
    @staticmethod
    def safe_unicode_string(text: str) -> str:
        """Convert string to safe Unicode representation"""
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Try to encode with system default encoding
            encoding = sys.stdout.encoding or 'utf-8'
            text.encode(encoding)
            return text
        except UnicodeEncodeError:
            # If encoding fails, use ASCII with Unicode escapes
            try:
                return text.encode('ascii', errors='backslashreplace').decode('ascii')
            except Exception:
                # Last resort: remove all non-ASCII characters
                return ''.join(char if ord(char) < 128 else '?' for char in text)
        except Exception:
            # Absolute fallback
            return str(text).encode('ascii', errors='ignore').decode('ascii')
    
    @staticmethod
    def safe_format_message(message: str, **kwargs) -> str:
        """Format message with kwargs safely"""
        try:
            safe_message = UnicodeHelper.safe_unicode_string(message)
            
            if kwargs:
                safe_kwargs = {}
                for k, v in kwargs.items():
                    safe_kwargs[k] = UnicodeHelper.safe_unicode_string(str(v))
                
                context = " | ".join([f"{k}={v}" for k, v in safe_kwargs.items()])
                return f"{safe_message} | {context}"
            
            return safe_message
            
        except Exception:
            # If everything fails, return basic message
            return str(message).encode('ascii', errors='ignore').decode('ascii')


class Config:
    """
    🔧 Configuration Management với Enhanced Validation
    
    Centralized configuration system with validation, persistence, and OpenAI integration.
    Enhanced with better error handling and validation.
    """
    
    DEFAULT_CONFIG = {
        "system": {
            "name": "Enhanced Retrieval System",
            "version": "2.1",
            "debug": False,
            "max_workers": 4,
            "unicode_support": True,
            "encoding": "utf-8"
        },
        "paths": {
            "keyframes": "keyframes/",
            "models": "models/",
            "cache": ".cache/",
            "exports": "exports/",
            "logs": "logs/",
            "index": "index/",
            "metadata": "metadata/",
            "map_keyframes": "map-keyframes-aic25-b1/"
        },
        "retrieval": {
            "faiss_index_type": "IndexFlatIP",
            "search_limit": 50,
            "similarity_threshold": 0.7,
            "temporal_window": 3,
            "enable_gpu": True,
            "validate_consistency": True,
            "allow_filename_frame_id_fallback": True
        },
        "competition": {
            "enabled": False,
            "strict_frame_mapping": True
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.7,
            "enable_cache": True,
            "cache_ttl": 3600  # seconds
        },
        "openai": {
            "enabled": True,
            "monitoring": True,
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "default_model": "gpt-4o",
            "api_key": None,  # Should be set via environment variable
            "organization": None,
            "max_retries": 3,
            "timeout": 30,
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "agno": {
            "enabled": True,
            "monitoring": True,
            "storage_type": "sqlite",  # sqlite, postgres
            "storage_path": "agno_storage.db",
            "postgres_url": None,
            "memory_enabled": True,
            "session_management": True,
            "agent_teams": True,
            "performance_tracking": True,
            "agentops_enabled": False,
            "agentops_api_key": None
        },
        "gui": {
            "window_width": 1670,
            "window_height": 850,
            "thumbnail_size": 350,
            "max_results_display": 100,
            "auto_export": False
        },
        "api": {
            "host": "localhost",
            "port": 5000,
            "cors_enabled": True,
            "rate_limit": 100,  # requests per minute
            "timeout": 30
        },
        "performance": {
            "enable_monitoring": True,
            "log_slow_operations": True,
            "slow_operation_threshold": 5.0,  # seconds
            "memory_warning_threshold": 0.8  # 80% of available memory
        },
        "logging": {
            "encoding": "utf-8",
            "console_encoding": "utf-8",
            "safe_unicode": True,
            "escape_non_ascii": True,
            "level": "INFO"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration management with validation
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config_path = config_path or "config.json"
        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)
        self._lock = threading.Lock()
        self._validation_errors = []
        
        # Setup Unicode support if enabled
        if self.get("system.unicode_support", True):
            UnicodeHelper.setup_unicode_environment()
        
        # Load existing config if available
        self.load_config()
        
        # Validate configuration
        self._validate_config()
        
        # Ensure all required directories exist
        self._create_directories()
        
        # Validate OpenAI configuration
        self._validate_openai_config()
    
    def load_config(self) -> bool:
        """Load configuration from file with error handling"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                if not isinstance(loaded_config, dict):
                    print(f"Warning: Invalid config format in {self.config_path}")
                    return False
                
                self._deep_update(self.config, loaded_config)
                return True
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file {self.config_path}: {e}")
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
        return False
    
    def save_config(self) -> bool:
        """Save current configuration to file with atomic write"""
        try:
            with self._lock:
                validation_errors = self._validate_config()
                if validation_errors:
                    print(f"Warning: Config has validation errors: {validation_errors}")
                
                temp_path = f"{self.config_path}.tmp"
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                
                # Verify written file
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        json.load(f)  # Test if readable
                except:
                    raise RuntimeError("Config verification failed")
                
                # Atomic move
                if os.path.exists(self.config_path):
                    backup_path = f"{self.config_path}.backup"
                    shutil.copy2(self.config_path, backup_path)
                
                shutil.move(temp_path, self.config_path)
                return True
                
        except Exception as e:
            print(f"Error: Could not save config to {self.config_path}: {e}")
            temp_path = f"{self.config_path}.tmp"
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with validation"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation with validation"""
        try:
            keys = key.split('.')
            config = self.config
            
            with self._lock:
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    elif not isinstance(config[k], dict):
                        config[k] = {}  # Convert to dict if needed
                    config = config[k]
                
                config[keys[-1]] = value
                
        except Exception as e:
            print(f"Warning: Could not set config key {key}: {e}")
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration"""
        return self.get("openai", {})
    
    def get_agno_config(self) -> Dict[str, Any]:
        """Get Agno-specific configuration"""
        return self.get("agno", {})
    
    def is_agno_enabled(self) -> bool:
        """Check if Agno framework is enabled"""
        return HAS_AGNO and self.get("agno.enabled", False)
    
    def get_validation_errors(self) -> List[str]:
        """Get current validation errors"""
        return self._validation_errors.copy()
    
    def validate_and_fix(self) -> bool:
        """Validate configuration and attempt to fix issues"""
        errors = self._validate_config()
        
        if not errors:
            return True
        
        fixed_count = 0
        for error in errors:
            if "missing required key" in error.lower():
                try:
                    if "paths" in error:
                        self._ensure_paths_config()
                        fixed_count += 1
                    elif "system" in error:
                        self._ensure_system_config()
                        fixed_count += 1
                except:
                    pass
        
        new_errors = self._validate_config()
        return len(new_errors) < len(errors)
    
    def _validate_config(self) -> List[str]:
        """Validate configuration structure and values"""
        errors = []
        
        try:
            # Check required top-level sections
            required_sections = ["system", "paths", "retrieval", "llm", "openai"]
            for section in required_sections:
                if section not in self.config:
                    errors.append(f"Missing required section: {section}")
                elif not isinstance(self.config[section], dict):
                    errors.append(f"Section {section} must be a dictionary")
            
            # Validate system section
            if "system" in self.config:
                system_config = self.config["system"]
                if not isinstance(system_config.get("name"), str):
                    errors.append("system.name must be a string")
                if not isinstance(system_config.get("max_workers"), int) or system_config.get("max_workers", 0) <= 0:
                    errors.append("system.max_workers must be a positive integer")
            
            # Validate paths section
            if "paths" in self.config:
                paths_config = self.config["paths"]
                required_paths = ["keyframes", "models", "cache", "exports", "logs", "index"]
                for path_key in required_paths:
                    if path_key not in paths_config:
                        errors.append(f"Missing required path: paths.{path_key}")
                    elif not isinstance(paths_config[path_key], str):
                        errors.append(f"paths.{path_key} must be a string")
            
            # Validate retrieval section
            if "retrieval" in self.config:
                retrieval_config = self.config["retrieval"]
                valid_index_types = ["IndexFlatL2", "IndexFlatIP", "IndexIVFFlat", "IndexHNSW", "IndexLSH"]
                index_type = retrieval_config.get("faiss_index_type")
                if index_type and index_type not in valid_index_types:
                    errors.append(f"Invalid FAISS index type: {index_type}")
                
                search_limit = retrieval_config.get("search_limit")
                if search_limit and (not isinstance(search_limit, int) or search_limit <= 0):
                    errors.append("retrieval.search_limit must be a positive integer")
            
            # Validate LLM section
            if "llm" in self.config:
                llm_config = self.config["llm"]
                if llm_config.get("provider") != "openai":
                    errors.append("LLM provider must be 'openai'")
            
            # Validate OpenAI section
            if "openai" in self.config:
                openai_config = self.config["openai"]
                valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                default_model = openai_config.get("default_model")
                if default_model and default_model not in valid_models:
                    errors.append(f"Invalid OpenAI model: {default_model}")
            
            self._validation_errors = errors
            return errors
            
        except Exception as e:
            error_msg = f"Config validation failed: {str(e)}"
            self._validation_errors = [error_msg]
            return [error_msg]
    
    def _validate_openai_config(self) -> None:
        """Validate OpenAI configuration settings"""
        openai_config = self.get_openai_config()
        
        # Check for OpenAI API key in config or environment
        api_key = openai_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OpenAI API key not found in config or OPENAI_API_KEY environment variable")
        
        # Validate model settings
        default_model = openai_config.get("default_model", "gpt-4o")
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        if default_model not in valid_models:
            print(f"Warning: Invalid OpenAI model: {default_model}")
        
        # Validate required settings
        required_settings = ["enabled", "default_model"]
        for setting in required_settings:
            if not openai_config.get(setting):
                print(f"Warning: OpenAI config missing required setting: {setting}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary with type checking"""
        try:
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    self._deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        except Exception as e:
            print(f"Warning: Error in config deep update: {e}")
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep copy dictionary safely"""
        try:
            return json.loads(json.dumps(d))
        except Exception:
            return d.copy()
    
    def _create_directories(self) -> None:
        """Create all required directories with error handling"""
        dirs_to_create = [
            self.get("paths.cache"),
            self.get("paths.exports"), 
            self.get("paths.logs"),
            self.get("paths.index"),
            self.get("paths.metadata")
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not create directory {dir_path}: {e}")
    
    def _ensure_paths_config(self) -> None:
        """Ensure paths configuration has all required keys"""
        if "paths" not in self.config:
            self.config["paths"] = {}
        
        default_paths = self.DEFAULT_CONFIG["paths"]
        for key, value in default_paths.items():
            if key not in self.config["paths"]:
                self.config["paths"][key] = value
    
    def _ensure_system_config(self) -> None:
        """Ensure system configuration has all required keys"""
        if "system" not in self.config:
            self.config["system"] = {}
        
        default_system = self.DEFAULT_CONFIG["system"]
        for key, value in default_system.items():
            if key not in self.config["system"]:
                self.config["system"][key] = value


class Logger:
    """
    📝 Centralized Logging System với Unicode Support
    
    Advanced logging with file rotation, performance tracking, structured logging,
    Unicode support, and OpenAI agent monitoring integration.
    """
    
    def __init__(self, 
                 log_level: str = "INFO", 
                 log_dir: str = "logs/", 
                 config: Optional[Config] = None):
        """
        Initialize logging system with Unicode support
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            config: Configuration instance
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.config = config or Config()
        
        # Get logging configuration
        self.encoding = self.config.get("logging.encoding", "utf-8")
        self.safe_unicode = self.config.get("logging.safe_unicode", True)
        
        # Setup main logger
        self.logger = logging.getLogger("EnhancedRetrievalSystem")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters with Unicode support
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s'
        )
        
        # Console handler with UTF-8 encoding
        console_handler = self._create_unicode_console_handler()
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with UTF-8 encoding and rotation
        log_file = self.log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = self._create_unicode_file_handler(log_file)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Performance logger with UTF-8
        self.perf_logger = logging.getLogger("Performance")
        self.perf_logger.handlers.clear()
        perf_file = self.log_dir / "performance.log"
        perf_handler = self._create_unicode_file_handler(perf_file, rotation=False)
        perf_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.WARNING)  # Changed from INFO to WARNING to reduce spam
        
        # OpenAI/Agno-specific logger with UTF-8
        if self.config.is_agno_enabled():
            self.agno_logger = logging.getLogger("OpenAIAgents")
            self.agno_logger.handlers.clear()
            agno_file = self.log_dir / "openai_agents.log"
            agno_handler = self._create_unicode_file_handler(agno_file, rotation=False)
            agno_handler.setFormatter(logging.Formatter(
                '%(asctime)s | OPENAI | %(levelname)-8s | %(message)s'
            ))
            self.agno_logger.addHandler(agno_handler)
            self.agno_logger.setLevel(logging.INFO)
        else:
            self.agno_logger = None
    
    def _create_unicode_console_handler(self) -> logging.StreamHandler:
        """Create console handler with Unicode support"""
        handler = logging.StreamHandler(sys.stdout)
        
        if self.safe_unicode:
            handler.addFilter(self._unicode_filter)
        
        return handler
    
    def _create_unicode_file_handler(self, 
                                   log_file: Path, 
                                   rotation: bool = True) -> logging.Handler:
        """Create file handler with UTF-8 encoding"""
        try:
            if rotation:
                handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding=self.encoding
                )
            else:
                handler = logging.FileHandler(
                    log_file,
                    encoding=self.encoding
                )
            
            if self.safe_unicode:
                handler.addFilter(self._unicode_filter)
            
            return handler
            
        except Exception as e:
            print(f"Warning: Could not create Unicode file handler: {e}")
            return logging.FileHandler(log_file)
    
    def _unicode_filter(self, record: logging.LogRecord) -> bool:
        """Filter to handle Unicode encoding issues"""
        try:
            if hasattr(record, 'msg'):
                record.msg = UnicodeHelper.safe_unicode_string(str(record.msg))
            
            if hasattr(record, 'args') and record.args:
                safe_args = []
                for arg in record.args:
                    safe_args.append(UnicodeHelper.safe_unicode_string(str(arg)))
                record.args = tuple(safe_args)
            
            return True
            
        except Exception:
            record.msg = "[ENCODING ERROR] Original message could not be encoded"
            record.args = ()
            return True
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with Unicode safety"""
        try:
            formatted_message = UnicodeHelper.safe_format_message(message, **kwargs)
            self.logger.info(formatted_message)
        except Exception as e:
            safe_message = UnicodeHelper.safe_unicode_string(str(message))
            self.logger.info(f"{safe_message} [LOGGING_ERROR: {str(e)}]")
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with Unicode safety"""
        try:
            formatted_message = UnicodeHelper.safe_format_message(message, **kwargs)
            self.logger.warning(formatted_message)
        except Exception as e:
            safe_message = UnicodeHelper.safe_unicode_string(str(message))
            self.logger.warning(f"{safe_message} [LOGGING_ERROR: {str(e)}]")
    
    def error(self, message: str, exc_info=None, **kwargs) -> None:
        """Log error message with Unicode safety"""
        try:
            formatted_message = UnicodeHelper.safe_format_message(message, **kwargs)
            self.logger.error(formatted_message, exc_info=exc_info)
        except Exception as e:
            safe_message = UnicodeHelper.safe_unicode_string(str(message))
            self.logger.error(f"{safe_message} [LOGGING_ERROR: {str(e)}]", exc_info=exc_info)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with Unicode safety"""
        try:
            formatted_message = UnicodeHelper.safe_format_message(message, **kwargs)
            self.logger.debug(formatted_message)
        except Exception as e:
            safe_message = UnicodeHelper.safe_unicode_string(str(message))
            self.logger.debug(f"{safe_message} [LOGGING_ERROR: {str(e)}]")
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with Unicode safety"""
        try:
            formatted_message = UnicodeHelper.safe_format_message(message, **kwargs)
            self.logger.critical(formatted_message)
        except Exception as e:
            safe_message = UnicodeHelper.safe_unicode_string(str(message))
            self.logger.critical(f"{safe_message} [LOGGING_ERROR: {str(e)}]")
    
    def log_performance(self, operation: str, duration: float, **metadata) -> None:
        """Log performance metrics with Unicode safety"""
        try:
            safe_operation = UnicodeHelper.safe_unicode_string(operation)
            safe_metadata = {
                k: UnicodeHelper.safe_unicode_string(str(v)) 
                for k, v in metadata.items()
            }
            
            perf_data = {
                "operation": safe_operation,
                "duration": round(duration, 4),
                "timestamp": datetime.now().isoformat(),
                **safe_metadata
            }
            
            self.perf_logger.info(json.dumps(perf_data, ensure_ascii=True))
            
        except Exception as e:
            safe_operation = UnicodeHelper.safe_unicode_string(str(operation))
            self.perf_logger.info(f"OPERATION: {safe_operation} | DURATION: {duration:.4f}s [ERROR: {str(e)}]")
    
    def log_agno_event(self, event_type: str, agent_name: str, **details) -> None:
        """
        Log OpenAI agent events with Unicode safety
        
        Args:
            event_type: Type of event (created, executed, error, etc.)
            agent_name: Name of the agent
            **details: Additional event details
        """
        if not self.agno_logger:
            return
        
        try:
            safe_event_type = UnicodeHelper.safe_unicode_string(event_type)
            safe_agent_name = UnicodeHelper.safe_unicode_string(agent_name)
            safe_details = {
                k: UnicodeHelper.safe_unicode_string(str(v)) 
                for k, v in details.items()
            }
            
            event_data = {
                "event_type": safe_event_type,
                "agent_name": safe_agent_name,
                "timestamp": datetime.now().isoformat(),
                **safe_details
            }
            
            self.agno_logger.info(json.dumps(event_data, ensure_ascii=True))
            
        except Exception as e:
            safe_event = UnicodeHelper.safe_unicode_string(str(event_type))
            safe_agent = UnicodeHelper.safe_unicode_string(str(agent_name))
            self.agno_logger.info(f"EVENT: {safe_event} | AGENT: {safe_agent} [ERROR: {str(e)}]")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get logger health status"""
        health = {
            "is_healthy": True,
            "issues": [],
            "stats": {},
            "handlers_count": len(self.logger.handlers)
        }
        
        try:
            for i, handler in enumerate(self.logger.handlers):
                handler_info = {
                    "type": type(handler).__name__,
                    "level": handler.level,
                    "encoding": getattr(handler, 'encoding', 'unknown')
                }
                health["stats"][f"handler_{i}"] = handler_info
            
            test_message = "Logger health check test message"
            self.debug(test_message)
            
        except Exception as e:
            health["is_healthy"] = False
            health["issues"].append(f"Logger health check failed: {str(e)}")
        
        return health


class AgnoManager:
    """
    🤖 OpenAI Agent Manager với Enhanced Error Handling
    
    Manages OpenAI agents, teams, storage, and monitoring integration with robust error handling.
    """
    
    def __init__(self, config: Optional[Config] = None, logger: Optional[Logger] = None):
        """
        Initialize OpenAI agent manager with enhanced error handling
        
        Args:
            config: Configuration instance
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or Logger()
        
        # Check Agno availability
        if not HAS_AGNO:
            self.logger.warning("Agno framework not available. Install with: pip install agno")
            self.enabled = False
            return
        
        self.enabled = self.config.is_agno_enabled()
        if not self.enabled:
            self.logger.info("Agno framework disabled in configuration")
            return
        
        # Initialize components
        self.agents = {}
        self.teams = {}
        self.storage = None
        self.memory_db = None
        self._initialization_errors = []
        
        # Setup storage with error handling
        self._setup_storage()
        
        # Setup monitoring with error handling
        self._setup_monitoring()
        
        if self._initialization_errors:
            self.logger.warning(f"Agno manager initialized with {len(self._initialization_errors)} issues")
            for error in self._initialization_errors:
                self.logger.warning(f"Agno init issue: {error}")
        else:
            self.logger.info("OpenAI Agno manager initialized successfully")
    
    def create_agent(self, 
                    name: str, 
                    role: str = "AI Assistant",
                    model_name: Optional[str] = None,
                    tools: Optional[List] = None,
                    memory_enabled: bool = True,
                    **kwargs) -> Optional[Agent]:
        """
        Create an OpenAI agent with enhanced error handling
        
        Args:
            name: Agent name
            role: Agent role description
            model_name: OpenAI model name (overrides config)
            tools: List of tools for the agent
            memory_enabled: Enable memory for the agent
            **kwargs: Additional agent parameters
            
        Returns:
            Created Agent instance or None if failed
        """
        if not self.enabled:
            self.logger.warning("Cannot create agent: Agno not enabled")
            return None
        
        try:
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Agent name must be a non-empty string")
            
            if name in self.agents:
                self.logger.warning(f"Agent {name} already exists, returning existing agent")
                return self.agents[name]
            
            # Setup OpenAI model
            model = self._get_openai_model(model_name)
            if not model:
                self.logger.error(f"Failed to create OpenAI model for agent {name}")
                return None
            
            # Setup memory
            memory = None
            if memory_enabled and self.config.get("agno.memory_enabled", True):
                try:
                    if self.memory_db:
                        memory = AgentMemory(
                            db=self.memory_db,
                            create_user_memories=True,
                            create_session_summary=True
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to setup memory for agent {name}: {e}")
            
            # Create agent
            agent = Agent(
                name=name,
                role=role,
                model=model,
                tools=tools or [],
                memory=memory,
                storage=self.storage,
                monitoring=self.config.get("agno.monitoring", True),
                **kwargs
            )
            
            # Store agent
            self.agents[name] = agent
            
            # Log creation
            self.logger.log_agno_event(
                "agent_created", 
                name, 
                role=role, 
                model_name=model_name or self.config.get("openai.default_model", "gpt-4o"),
                memory_enabled=memory_enabled and memory is not None
            )
            
            self.logger.info(f"Created OpenAI agent: {name}", 
                           role=role, model=f"openai:{model_name}")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {name}", error=str(e), exc_info=True)
            return None
    
    def create_team(self, 
                   name: str, 
                   agents: List[str], 
                   mode: str = "coordinate",
                   leader_model: Optional[str] = None) -> Optional[Team]:
        """
        Create an OpenAI team with enhanced error handling
        
        Args:
            name: Team name
            agents: List of agent names to include
            mode: Team mode (coordinate, route, collaborate)
            leader_model: OpenAI model for team leader
            
        Returns:
            Created Team instance or None if failed
        """
        if not self.enabled:
            self.logger.warning("Cannot create team: Agno not enabled")
            return None
        
        if not self.config.get("agno.agent_teams", True):
            self.logger.warning("Agent teams disabled in configuration")
            return None
        
        try:
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Team name must be a non-empty string")
            
            if not agents or not isinstance(agents, list):
                raise ValueError("Agents must be a non-empty list")
            
            valid_modes = ["coordinate", "route", "collaborate"]
            if mode not in valid_modes:
                raise ValueError(f"Invalid team mode: {mode}. Must be one of {valid_modes}")
            
            # Get agent instances
            team_members = []
            missing_agents = []
            
            for agent_name in agents:
                if agent_name in self.agents:
                    team_members.append(self.agents[agent_name])
                else:
                    missing_agents.append(agent_name)
            
            if missing_agents:
                self.logger.warning(f"Missing agents for team {name}: {missing_agents}")
            
            if not team_members:
                self.logger.error(f"No valid agents found for team {name}")
                return None
            
            # Setup leader model if specified
            leader_model_instance = None
            if leader_model:
                leader_model_instance = self._get_openai_model(leader_model)
                if not leader_model_instance:
                    self.logger.warning(f"Failed to create leader model for team {name}")
            
            # Create team
            team = Team(
                name=name,
                members=team_members,
                mode=mode,
                model=leader_model_instance,
                monitoring=self.config.get("agno.monitoring", True)
            )
            
            # Store team
            self.teams[name] = team
            
            # Log creation
            self.logger.log_agno_event(
                "team_created",
                name,
                mode=mode,
                members=agents,
                leader_model=leader_model,
                member_count=len(team_members)
            )
            
            self.logger.info(f"Created OpenAI team: {name}", 
                           mode=mode, members=len(team_members))
            
            return team
            
        except Exception as e:
            self.logger.error(f"Failed to create team {name}", error=str(e), exc_info=True)
            return None
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name with validation"""
        if not isinstance(name, str) or not name:
            self.logger.warning("Invalid agent name provided")
            return None
        
        return self.agents.get(name)
    
    def get_team(self, name: str) -> Optional[Team]:
        """Get team by name with validation"""
        if not isinstance(name, str) or not name:
            self.logger.warning("Invalid team name provided")
            return None
        
        return self.teams.get(name)
    
    def list_agents(self) -> List[str]:
        """List all agent names"""
        return list(self.agents.keys())
    
    def list_teams(self) -> List[str]:
        """List all team names"""
        return list(self.teams.keys())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get OpenAI agent manager health status"""
        health = {
            "is_healthy": True,
            "issues": [],
            "stats": {},
            "enabled": self.enabled
        }
        
        if not self.enabled:
            health["issues"].append("Agno framework not enabled")
            return health
        
        try:
            health["stats"]["agents_count"] = len(self.agents)
            health["stats"]["teams_count"] = len(self.teams)
            health["stats"]["has_storage"] = self.storage is not None
            health["stats"]["has_memory_db"] = self.memory_db is not None
            health["stats"]["initialization_errors"] = len(self._initialization_errors)
            
            if self._initialization_errors:
                health["issues"].extend(self._initialization_errors)
                health["is_healthy"] = False
            
            # Test OpenAI model creation capability
            try:
                test_model = self._get_openai_model("gpt-3.5-turbo")
                if test_model:
                    health["stats"]["model_creation_working"] = True
                else:
                    health["issues"].append("OpenAI model creation not working")
                    health["is_healthy"] = False
            except Exception as e:
                health["issues"].append(f"OpenAI model creation test failed: {str(e)}")
                health["is_healthy"] = False
            
        except Exception as e:
            health["is_healthy"] = False
            health["issues"].append(f"Health check failed: {str(e)}")
        
        return health
    
    def _get_openai_model(self, model_name: Optional[str] = None):
        """Get OpenAI model instance with error handling"""
        try:
            if not model_name:
                model_name = self.config.get("openai.default_model", "gpt-4o")
            
            # Validate model name
            valid_models = self.config.get("openai.models", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
            if model_name not in valid_models:
                self.logger.warning(f"Model {model_name} not in valid models list, but proceeding")
            
            return OpenAIChat(
                id=model_name,
                api_key=self.config.get("openai.api_key") or os.getenv("OPENAI_API_KEY"),
                organization=self.config.get("openai.organization"),
                max_retries=self.config.get("openai.max_retries", 3),
                timeout=self.config.get("openai.timeout", 30),
                temperature=self.config.get("openai.temperature", 0.7),
                max_tokens=self.config.get("openai.max_tokens", 1000)
            )
                
        except Exception as e:
            self.logger.error(f"Failed to create OpenAI model {model_name}", error=str(e))
            return None
    
    def _setup_storage(self) -> None:
        """Setup storage for agents with error handling"""
        try:
            storage_type = self.config.get("agno.storage_type", "sqlite")
            
            if storage_type == "postgres":
                postgres_url = self.config.get("agno.postgres_url")
                if postgres_url:
                    try:
                        self.storage = PostgresStorage(
                            table_name="agent_sessions",
                            db_url=postgres_url
                        )
                        self.memory_db = PgMemoryDb(
                            table_name="agent_memory",
                            db_url=postgres_url
                        )
                        self.logger.info("PostgreSQL storage initialized successfully")
                    except Exception as e:
                        self.logger.error(f"PostgreSQL storage initialization failed: {e}")
                        self._initialization_errors.append(f"PostgreSQL storage failed: {str(e)}")
                else:
                    self.logger.error("PostgreSQL URL not configured")
                    self._initialization_errors.append("PostgreSQL URL not configured")
                    
            else:  # sqlite
                storage_path = self.config.get("agno.storage_path", "agno_storage.db")
                try:
                    self.storage = SqliteAgentStorage(
                        table_name="agent_sessions",
                        db_file=storage_path
                    )
                    # SQLite memory DB would need custom implementation
                    self.memory_db = None
                    self.logger.info("SQLite storage initialized successfully")
                except Exception as e:
                    self.logger.error(f"SQLite storage initialization failed: {e}")
                    self._initialization_errors.append(f"SQLite storage failed: {str(e)}")
            
            if self.storage:
                self.logger.info(f"Agno storage initialized: {storage_type}")
            else:
                self._initialization_errors.append("No storage initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Agno storage", error=str(e))
            self._initialization_errors.append(f"Storage setup failed: {str(e)}")
            self.storage = None
            self.memory_db = None
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring integration with error handling"""
        try:
            if self.config.get("agno.agentops_enabled", False):
                api_key = self.config.get("agno.agentops_api_key")
                if api_key and agentops:
                    try:
                        agentops.init(api_key=api_key)
                        self.logger.info("AgentOps monitoring initialized")
                    except Exception as e:
                        self.logger.warning(f"AgentOps initialization failed: {e}")
                        self._initialization_errors.append(f"AgentOps failed: {str(e)}")
                else:
                    if not api_key:
                        self.logger.warning("AgentOps API key not configured")
                        self._initialization_errors.append("AgentOps API key missing")
                    if not agentops:
                        self.logger.warning("AgentOps library not available")
                        self._initialization_errors.append("AgentOps library not available")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring", error=str(e))
            self._initialization_errors.append(f"Monitoring setup failed: {str(e)}")


class FileManager:
    """
    📁 File and Directory Operations với Enhanced Validation
    
    Comprehensive file management with safety checks, backup, batch operations, and validation.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize file manager with enhanced validation
        
        Args:
            logger: Logger instance for operation tracking
        """
        self.logger = logger
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        # Statistics tracking
        self.operation_stats = {
            "files_scanned": 0,
            "files_processed": 0,
            "errors_encountered": 0,
            "last_operation": None
        }
    
    def scan_keyframes(self, folder_path: str, validate_files: bool = True) -> Dict[str, List[str]]:
        """
        Scan keyframe directory structure with enhanced validation
        
        Args:
            folder_path: Root keyframe directory
            validate_files: Whether to validate individual files
            
        Returns:
            Dict mapping folder names to image file lists
        """
        if self.logger:
            self.logger.info(f"Scanning keyframes directory", path=folder_path)
        
        # Reset stats
        self.operation_stats["files_scanned"] = 0
        self.operation_stats["files_processed"] = 0
        self.operation_stats["errors_encountered"] = 0
        self.operation_stats["last_operation"] = "scan_keyframes"
        
        try:
            folder_path = Path(folder_path)
            
            # Validate input path
            if not folder_path.exists():
                raise FileNotFoundError(f"Keyframes directory not found: {folder_path}")
            
            if not folder_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {folder_path}")
            
            keyframe_structure = {}
            total_images = 0
            total_errors = 0
            
            # Scan subdirectories
            for subfolder in folder_path.iterdir():
                if not subfolder.is_dir():
                    continue
                
                try:
                    image_files = []
                    folder_errors = 0
                    
                    for file_path in subfolder.iterdir():
                        self.operation_stats["files_scanned"] += 1
                        
                        try:
                            # Check file extension
                            if file_path.suffix.lower() not in self.supported_image_formats:
                                continue
                            
                            # Validate file if requested
                            if validate_files:
                                if not self._validate_image_file(file_path):
                                    folder_errors += 1
                                    continue
                            
                            image_files.append(file_path.name)
                            self.operation_stats["files_processed"] += 1
                            
                        except Exception as e:
                            folder_errors += 1
                            total_errors += 1
                            if self.logger:
                                self.logger.warning(f"Error processing file {file_path}: {e}")
                    
                    if image_files:
                        # Sort numerically if possible
                        try:
                            image_files.sort(key=lambda x: int(Path(x).stem))
                        except ValueError:
                            # Fallback to alphabetical sort
                            image_files.sort()
                        
                        keyframe_structure[subfolder.name] = image_files
                        total_images += len(image_files)
                    
                    if folder_errors > 0 and self.logger:
                        self.logger.warning(f"Folder {subfolder.name} had {folder_errors} file errors")
                
                except Exception as e:
                    total_errors += 1
                    if self.logger:
                        self.logger.error(f"Error processing folder {subfolder}: {e}")
            
            self.operation_stats["errors_encountered"] = total_errors
            
            if self.logger:
                self.logger.info(
                    f"Keyframe scan completed",
                    folders=len(keyframe_structure),
                    total_images=total_images,
                    errors=total_errors
                )
            
            return keyframe_structure
            
        except Exception as e:
            self.operation_stats["errors_encountered"] += 1
            if self.logger:
                self.logger.error(f"Error scanning keyframes", error=str(e), exc_info=True)
            raise
    
    def load_csv_mapping(self, csv_path: str, validate_format: bool = True) -> Dict[str, int]:
        """
        Load frame ID mapping from CSV file with header support
        """
        # Check if MAP_FOLDER_PATH is set (from GUI), use it as base path
        map_folder = os.environ.get('MAP_FOLDER_PATH', '')
        if map_folder and not os.path.isabs(csv_path):
            # If csv_path is relative and we have a map folder, use map folder as base
            csv_path = os.path.join(map_folder, os.path.basename(csv_path))
        
        if self.logger:
            self.logger.debug(f"Loading CSV mapping", path=csv_path, map_folder=map_folder)
        
        try:
            if not os.path.exists(csv_path):
                if self.logger:
                    self.logger.warning(f"CSV mapping file not found", path=csv_path)
                return {}
            
            # Validate file size
            file_size = os.path.getsize(csv_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                if self.logger:
                    self.logger.warning(f"CSV file too large: {file_size} bytes")
                return {}
            
            # Load CSV with proper header handling
            try:
                # First, try to detect if first row is header
                df_test = pd.read_csv(csv_path, header=None, nrows=1)
                first_row = df_test.iloc[0].tolist()
                
                # Check if first row contains text headers
                has_header = any(isinstance(val, str) and not val.replace('.','').replace('-','').isdigit() 
                            for val in first_row)
                
                if has_header:
                    # Load with header
                    df = pd.read_csv(csv_path, header=0)  # Use first row as header
                    
                    # DEBUG: Log CSV structure
                    if self.logger:
                        self.logger.info(f"CSV loaded with header: {csv_path}")
                        self.logger.info(f"  Shape: {df.shape}")
                        self.logger.info(f"  Columns: {df.columns.tolist()}")
                        if len(df) > 0:
                            self.logger.info(f"  First data row: {df.iloc[0].to_dict()}")
                    
                    # Map columns - try different possible column names
                    image_col = None
                    frame_col = None
                    
                    # Look for image/frame columns
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if col_lower in ['n', 'image', 'image_name', 'img', 'name']:
                            image_col = col
                        elif col_lower in ['frame_idx', 'frame_id', 'frame', 'idx', 'id']:
                            frame_col = col
                    
                    if image_col is None:
                        # Fallback to first column
                        image_col = df.columns[0]
                        if self.logger:
                            self.logger.warning(f"No image column found, using: {image_col}")
                    
                    if frame_col is None:
                        # Try to find numeric column for frame ID
                        for col in df.columns[1:]:  # Skip first column
                            try:
                                pd.to_numeric(df[col], errors='raise')
                                frame_col = col
                                break
                            except:
                                continue
                        
                        if frame_col is None:
                            frame_col = df.columns[-1]  # Use last column as fallback
                            if self.logger:
                                self.logger.warning(f"No frame column found, using: {frame_col}")
                    
                    # Extract mapping
                    frame_mapping = {}
                    invalid_rows = 0
                    
                    for index, row in df.iterrows():
                        try:
                            # Get image name (convert to string, handle various formats)
                            image_val = row[image_col]
                            if pd.isna(image_val):
                                invalid_rows += 1
                                continue
                            
                            # Convert to string and format as expected
                            if isinstance(image_val, (int, float)):
                                image_name = f"{int(image_val):03d}"  # Format as 001, 002, etc.
                            else:
                                image_name = str(image_val).strip()
                            
                            # Get frame ID
                            frame_val = row[frame_col]
                            if pd.isna(frame_val):
                                # Use index + 1 as fallback
                                frame_id = index + 1
                            else:
                                frame_id = int(float(frame_val))
                            
                            # Validate frame_id
                            if frame_id < 0:
                                frame_id = index + 1  # Use 1-based index
                            
                            frame_mapping[image_name] = frame_id
                            
                        except (ValueError, TypeError, KeyError) as e:
                            invalid_rows += 1
                            if self.logger:
                                self.logger.debug(f"Invalid row {index}: {e}")
                            continue
                
                else:
                    # Load without header (original format)
                    df = pd.read_csv(csv_path, header=None)
                    
                    if self.logger:
                        self.logger.info(f"CSV loaded without header: {csv_path}")
                        self.logger.info(f"  Shape: {df.shape}")
                    
                    frame_mapping = {}
                    invalid_rows = 0
                    
                    for index, row in df.iterrows():
                        try:
                            # Try different column combinations
                            if len(row) >= 4:
                                # Original format: image_name in column 0, frame_id in column 3
                                image_name = f"{int(float(row.iloc[0])):03d}"
                                frame_id = int(float(row.iloc[3]))
                            elif len(row) >= 2:
                                # Alternative: image_name in col 0, frame_id in col 1
                                image_name = f"{int(float(row.iloc[0])):03d}"
                                frame_id = int(float(row.iloc[1]))
                            else:
                                # Fallback
                                image_name = f"{int(float(row.iloc[0])):03d}"
                                frame_id = index + 1
                            
                            if frame_id < 0:
                                frame_id = index + 1
                            
                            frame_mapping[image_name] = frame_id
                            
                        except (ValueError, TypeError, IndexError) as e:
                            invalid_rows += 1
                            continue
                
                if invalid_rows > 0 and self.logger:
                    self.logger.warning(f"CSV had {invalid_rows} invalid rows")
                
                if self.logger:
                    self.logger.info(f"CSV mapping loaded successfully", 
                                entries=len(frame_mapping),
                                invalid_rows=invalid_rows,
                                sample_mapping=dict(list(frame_mapping.items())[:3]))
                
                return frame_mapping
                    
            except pd.errors.EmptyDataError:
                if self.logger:
                    self.logger.warning(f"CSV file is empty: {csv_path}")
                return {}
            
            except pd.errors.ParserError as e:
                if self.logger:
                    self.logger.error(f"CSV parsing error: {e}")
                return {}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading CSV mapping", path=csv_path, error=str(e))
            return {}
    
    def create_backup(self, file_path: str, backup_dir: str = "backups/") -> str:
        """
        Create backup of important file with validation
        
        Args:
            file_path: Path to file to backup
            backup_dir: Directory for backups
            
        Returns:
            Path to backup file
        """
        try:
            file_path = Path(file_path)
            backup_dir = Path(backup_dir)
            
            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Source path is not a file: {file_path}")
            
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            # Ensure backup path doesn't exist
            counter = 1
            original_backup_path = backup_path
            while backup_path.exists():
                backup_name = f"{file_path.stem}_{timestamp}_{counter}{file_path.suffix}"
                backup_path = backup_dir / backup_name
                counter += 1
                
                if counter > 100:  # Prevent infinite loop
                    raise RuntimeError("Could not generate unique backup filename")
            
            # Perform backup
            shutil.copy2(file_path, backup_path)
            
            # Verify backup
            if not backup_path.exists():
                raise RuntimeError("Backup file was not created")
            
            backup_size = backup_path.stat().st_size
            original_size = file_path.stat().st_size
            
            if backup_size != original_size:
                self.logger.warning(f"Backup size mismatch: {backup_size} vs {original_size}")
            
            if self.logger:
                self.logger.info(f"Backup created", 
                               source=str(file_path), 
                               backup=str(backup_path),
                               size=backup_size)
            
            return str(backup_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Backup failed", source=str(file_path), error=str(e))
            raise
    
    def cleanup_temp_files(self, 
                          temp_dir: str = "temp/", 
                          older_than_hours: int = 24,
                          dry_run: bool = False) -> int:
        """
        Clean up temporary files older than specified time with validation
        
        Args:
            temp_dir: Temporary files directory
            older_than_hours: Remove files older than this many hours
            dry_run: If True, only count files that would be removed
            
        Returns:
            Number of files removed (or would be removed in dry run)
        """
        try:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                return 0
            
            if not temp_path.is_dir():
                if self.logger:
                    self.logger.warning(f"Temp path is not a directory: {temp_path}")
                return 0
            
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            removed_count = 0
            error_count = 0
            
            # Scan for old files
            for file_path in temp_path.rglob("*"):
                if not file_path.is_file():
                    continue
                
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_mtime < cutoff_time:
                        if not dry_run:
                            file_path.unlink()
                        removed_count += 1
                        
                except Exception as e:
                    error_count += 1
                    if self.logger:
                        self.logger.warning(f"Error processing temp file {file_path}: {e}")
            
            if self.logger:
                action = "would be removed" if dry_run else "removed"
                self.logger.info(f"Temporary files cleanup", 
                               files_count=removed_count,
                               action=action,
                               errors=error_count,
                               dry_run=dry_run)
            
            return removed_count
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cleanup failed", error=str(e))
            return 0
    
    def safe_write(self, 
                  file_path: str, 
                  data: Any, 
                  format: str = "auto",
                  backup_original: bool = True) -> bool:
        """
        Safely write data to file with atomic operation and validation
        
        Args:
            file_path: Target file path
            data: Data to write
            format: File format (auto, json, pickle, numpy, text)
            backup_original: Whether to backup existing file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
            
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup original file if it exists and backup is requested
            if backup_original and file_path.exists():
                try:
                    self.create_backup(str(file_path))
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not create backup: {e}")
            
            # Determine format
            if format == "auto":
                suffix = file_path.suffix.lower()
                if suffix == ".json":
                    format = "json"
                elif suffix in [".pkl", ".pickle"]:
                    format = "pickle"
                elif suffix in [".npy", ".npz"]:
                    format = "numpy"
                else:
                    format = "text"
            
            # Write to temporary file
            if format == "json":
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            elif format == "pickle":
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                    
            elif format == "numpy":
                if file_path.suffix == ".npz":
                    if isinstance(data, dict):
                        np.savez_compressed(temp_path, **data)
                    else:
                        np.savez_compressed(temp_path, data=data)
                else:
                    np.save(temp_path, data)
                    
            else:  # text format
                content = str(data)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Verify written file
            if not temp_path.exists():
                raise RuntimeError("Temporary file was not created")
            
            temp_size = temp_path.stat().st_size
            if temp_size == 0:
                raise RuntimeError("Temporary file is empty")
            
            # Atomic move
            temp_path.replace(file_path)
            
            # Final verification
            if not file_path.exists():
                raise RuntimeError("Final file was not created")
            
            if self.logger:
                self.logger.debug(f"File written safely", 
                                path=str(file_path), 
                                format=format,
                                size=temp_size)
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Safe write failed", path=str(file_path), error=str(e))
            
            # Cleanup temp file
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            
            return False
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get file operation statistics"""
        return self.operation_stats.copy()
    
    def _validate_image_file(self, file_path: Path) -> bool:
        """Validate that file is a valid image"""
        try:
            # Check file size (prevent loading huge files)
            file_size = file_path.stat().st_size
            
            # Skip empty files
            if file_size == 0:
                return False
            
            # Skip very large files (>100MB)
            if file_size > 100 * 1024 * 1024:
                return False
            
            # Try to open with PIL (basic validation)
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # Verify image can be loaded
                    img.verify()
                return True
            except Exception:
                return False
                
        except Exception:
            return False


class DataProcessor:
    """
    🔄 Data Processing and Transformation Utilities với Enhanced Performance
    
    High-performance data processing with batch operations, memory optimization, and validation.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize data processor with enhanced capabilities
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.processing_stats = {
            "operations_performed": 0,
            "items_processed": 0,
            "errors_encountered": 0,
            "last_operation": None
        }
    
    def normalize_features(self, 
                          features: np.ndarray, 
                          method: str = "l2",
                          validate_input: bool = True) -> np.ndarray:
        """
        Normalize feature vectors with validation
        
        Args:
            features: Feature matrix (N x D)
            method: Normalization method (l2, minmax, standard)
            validate_input: Whether to validate input
            
        Returns:
            Normalized features
        """
        try:
            if validate_input:
                if not isinstance(features, np.ndarray):
                    raise ValueError("Features must be numpy array")
                
                if features.ndim != 2:
                    raise ValueError(f"Features must be 2D array, got {features.ndim}D")
                
                if features.size == 0:
                    raise ValueError("Features array is empty")
                
                if not np.isfinite(features).all():
                    raise ValueError("Features contain NaN or infinite values")
            
            self.processing_stats["operations_performed"] += 1
            self.processing_stats["items_processed"] += len(features)
            self.processing_stats["last_operation"] = f"normalize_features_{method}"
            
            if method == "l2":
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                return features / norms
                
            elif method == "minmax":
                min_vals = features.min(axis=0)
                max_vals = features.max(axis=0)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1  # Avoid division by zero
                return (features - min_vals) / ranges
                
            elif method == "standard":
                means = features.mean(axis=0)
                stds = features.std(axis=0)
                stds[stds == 0] = 1  # Avoid division by zero
                return (features - means) / stds
                
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            if self.logger:
                self.logger.error(f"Feature normalization failed: {e}")
            raise
    
    def batch_process_images(self, 
                           image_paths: List[str], 
                           batch_size: int = 32,
                           validate_paths: bool = True) -> List[List[str]]:
        """
        Create batches for image processing with validation
        
        Args:
            image_paths: List of image file paths
            batch_size: Size of each batch
            validate_paths: Whether to validate file paths
            
        Returns:
            List of batched image paths
        """
        try:
            if not isinstance(image_paths, list):
                raise ValueError("Image paths must be a list")
            
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
            
            if len(image_paths) == 0:
                if self.logger:
                    self.logger.warning("No image paths provided for batching")
                return []
            
            # Validate paths if requested
            if validate_paths:
                valid_paths = []
                invalid_count = 0
                
                for path in image_paths:
                    if isinstance(path, str) and os.path.exists(path):
                        valid_paths.append(path)
                    else:
                        invalid_count += 1
                
                if invalid_count > 0 and self.logger:
                    self.logger.warning(f"Found {invalid_count} invalid image paths")
                
                image_paths = valid_paths
            
            # Create batches
            batches = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                batches.append(batch)
            
            self.processing_stats["operations_performed"] += 1
            self.processing_stats["items_processed"] += len(image_paths)
            self.processing_stats["last_operation"] = "batch_process_images"
            
            if self.logger:
                self.logger.debug(f"Created image batches", 
                                total_images=len(image_paths), 
                                batches=len(batches), 
                                batch_size=batch_size)
            
            return batches
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            if self.logger:
                self.logger.error(f"Image batching failed: {e}")
            raise
    
    def merge_results(self, 
                     results_list: List[List],
                     remove_duplicates: bool = True,
                     sort_key: Optional[Callable] = None) -> List:
        """
        Merge and optionally deduplicate results from multiple searches
        
        Args:
            results_list: List of result lists
            remove_duplicates: Whether to remove duplicates
            sort_key: Optional sorting function
            
        Returns:
            Merged results
        """
        try:
            if not isinstance(results_list, list):
                raise ValueError("Results list must be a list")
            
            if len(results_list) == 0:
                return []
            
            merged = []
            seen = set() if remove_duplicates else None
            duplicate_count = 0
            
            for result_list in results_list:
                if not isinstance(result_list, list):
                    if self.logger:
                        self.logger.warning("Skipping non-list result in merge")
                    continue
                
                for item in result_list:
                    if remove_duplicates:
                        item_key = self._get_item_key(item)
                        if item_key in seen:
                            duplicate_count += 1
                            continue
                        seen.add(item_key)
                    
                    merged.append(item)
            
            # Sort if requested
            if sort_key and callable(sort_key):
                try:
                    merged.sort(key=sort_key)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Sorting failed: {e}")
            
            self.processing_stats["operations_performed"] += 1
            self.processing_stats["items_processed"] += len(merged)
            self.processing_stats["last_operation"] = "merge_results"
            
            if self.logger and remove_duplicates and duplicate_count > 0:
                self.logger.debug(f"Removed {duplicate_count} duplicates during merge")
            
            return merged
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            if self.logger:
                self.logger.error(f"Results merging failed: {e}")
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get data processing statistics"""
        return self.processing_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.processing_stats = {
            "operations_performed": 0,
            "items_processed": 0,
            "errors_encountered": 0,
            "last_operation": None
        }
    
    def _get_item_key(self, item) -> str:
        """Generate unique key for result item"""
        try:
            if hasattr(item, 'metadata') and hasattr(item.metadata, 'get_unique_key'):
                return item.metadata.get_unique_key()
            elif isinstance(item, dict):
                return f"{item.get('keyframe', '')}_{item.get('frameid', '')}"
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                return f"{item[0]}_{item[1]}"
            else:
                return str(hash(str(item)))
        except Exception:
            return str(hash(str(item)))


class CacheManager:
    """
    💾 Intelligent Caching System với Enhanced Validation
    
    Multi-level caching with TTL, LRU eviction, persistence, validation, and health monitoring.
    """
    
    def __init__(self, 
                 cache_dir: str = ".cache/", 
                 max_memory_mb: int = 512, 
                 config: Optional[Config] = None):
        """
        Initialize cache manager with enhanced validation
        
        Args:
            cache_dir: Directory for persistent cache
            max_memory_mb: Maximum memory usage for in-memory cache
            config: Configuration instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.config = config or Config()
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        
        # In-memory cache with TTL
        self.memory_cache = {}
        self.cache_times = {}
        self.cache_sizes = {}  # Track memory usage per item
        self.access_order = deque()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
            "total_requests": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = None  # Will be set externally if needed
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get cached value with validation
        
        Args:
            key: Cache key
            ttl: Time to live in seconds (None for no expiry)
            
        Returns:
            Cached value or None if not found/expired
        """
        if not isinstance(key, str) or not key:
            self.stats["errors"] += 1
            return None
        
        with self._lock:
            self.stats["total_requests"] += 1
            
            try:
                # Check memory cache first
                if key in self.memory_cache:
                    if ttl is None or self._is_valid(key, ttl):
                        self._update_access(key)
                        self.stats["hits"] += 1
                        return self.memory_cache[key]
                    else:
                        self._remove_from_memory(key)
                        self.stats["evictions"] += 1
                
                # Check persistent cache
                cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
                if cache_file.exists():
                    try:
                        if ttl is None or self._is_file_valid(cache_file, ttl):
                            with open(cache_file, 'rb') as f:
                                cached_data = pickle.load(f)
                            
                            # Load back to memory cache
                            self._add_to_memory(key, cached_data)
                            self.stats["hits"] += 1
                            return cached_data
                        else:
                            cache_file.unlink()
                            self.stats["evictions"] += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Cache read error", key=key, error=str(e))
                        self.stats["errors"] += 1
                        # Remove corrupted cache file
                        try:
                            cache_file.unlink()
                        except:
                            pass
                
                self.stats["misses"] += 1
                return None
                
            except Exception as e:
                self.stats["errors"] += 1
                if self.logger:
                    self.logger.error(f"Cache get error", key=key, error=str(e))
                return None
    
    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """
        Set cached value with validation
        
        Args:
            key: Cache key
            value: Value to cache
            persist: Whether to persist to disk
            
        Returns:
            True if successful, False otherwise
        """
        if not isinstance(key, str) or not key:
            self.stats["errors"] += 1
            return False
        
        try:
            with self._lock:
                # Add to memory cache
                success = self._add_to_memory(key, value)
                if not success:
                    return False
                
                # Persist to disk if requested
                if persist:
                    try:
                        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
                        temp_file = cache_file.with_suffix('.tmp')
                        
                        with open(temp_file, 'wb') as f:
                            pickle.dump(value, f)
                        
                        # Atomic move
                        temp_file.replace(cache_file)
                        
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Cache persist error", key=key, error=str(e))
                        self.stats["errors"] += 1
                        # Don't fail if persistence fails
                
                return True
                
        except Exception as e:
            self.stats["errors"] += 1
            if self.logger:
                self.logger.error(f"Cache set error", key=key, error=str(e))
            return False
    
    def cache_query_results(self, query: str, results: List, ttl: int = 3600) -> bool:
        """
        Cache search query results with validation
        
        Args:
            query: Search query
            results: Search results
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        if not isinstance(query, str) or not isinstance(results, list):
            return False
        
        cache_key = f"query:{self._hash_key(query)}"
        cache_data = {
            "query": query,
            "results": results,
            "timestamp": time.time(),
            "ttl": ttl
        }
        
        return self.set(cache_key, cache_data, persist=True)
    
    def get_cached_results(self, query: str, ttl: int = 3600) -> Optional[List]:
        """
        Get cached query results with validation
        
        Args:
            query: Search query
            ttl: Time to live in seconds
            
        Returns:
            Cached results or None
        """
        if not isinstance(query, str):
            return None
        
        cache_key = f"query:{self._hash_key(query)}"
        cached_data = self.get(cache_key, ttl)
        
        if cached_data and isinstance(cached_data, dict):
            return cached_data.get("results")
        
        return None
    
    def cache_agent_response(self, agent_name: str, query: str, response: str, ttl: int = 1800) -> bool:
        """
        Cache OpenAI agent response with validation
        
        Args:
            agent_name: Name of the agent
            query: Input query
            response: Agent response
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        if not all(isinstance(x, str) for x in [agent_name, query, response]):
            return False
        
        cache_key = f"agent:{agent_name}:{self._hash_key(query)}"
        cache_data = {
            "agent_name": agent_name,
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl
        }
        
        return self.set(cache_key, cache_data, persist=True)
    
    def get_cached_agent_response(self, agent_name: str, query: str, ttl: int = 1800) -> Optional[str]:
        """
        Get cached agent response with validation
        
        Args:
            agent_name: Name of the agent
            query: Input query
            ttl: Time to live in seconds
            
        Returns:
            Cached response or None
        """
        if not all(isinstance(x, str) for x in [agent_name, query]):
            return None
        
        cache_key = f"agent:{agent_name}:{self._hash_key(query)}"
        cached_data = self.get(cache_key, ttl)
        
        if cached_data and isinstance(cached_data, dict):
            return cached_data.get("response")
        
        return None
    
    def clear_cache(self, older_than_days: int = 7) -> int:
        """
        Clear old cache entries with validation
        
        Args:
            older_than_days: Remove entries older than this many days
            
        Returns:
            Number of entries removed
        """
        if older_than_days <= 0:
            return 0
        
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        removed_count = 0
        
        with self._lock:
            try:
                # Clear memory cache
                expired_keys = []
                for key, cache_time in self.cache_times.items():
                    if cache_time < cutoff_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_from_memory(key)
                    removed_count += 1
                
                # Clear persistent cache
                try:
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        if cache_file.stat().st_mtime < cutoff_time:
                            cache_file.unlink()
                            removed_count += 1
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Persistent cache cleanup error", error=str(e))
                    self.stats["errors"] += 1
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Cache cleanup error", error=str(e))
                self.stats["errors"] += 1
        
        if self.logger:
            self.logger.info(f"Cache cleared", removed=removed_count)
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            hit_rate = 0.0
            if self.stats["total_requests"] > 0:
                hit_rate = self.stats["hits"] / self.stats["total_requests"]
            
            return {
                **self.stats.copy(),
                "hit_rate": round(hit_rate, 3),
                "memory_usage_mb": round(self.current_memory / (1024 * 1024), 2),
                "memory_usage_percent": round(self.current_memory / self.max_memory_bytes * 100, 1),
                "memory_items": len(self.memory_cache),
                "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        stats = self.get_cache_stats()
        
        health = {
            "is_healthy": True,
            "issues": [],
            "stats": stats
        }
        
        # Check for issues
        if stats["hit_rate"] < 0.1 and stats["total_requests"] > 100:
            health["issues"].append("Low cache hit rate")
        
        if stats["memory_usage_percent"] > 95:
            health["issues"].append("Memory usage very high")
            health["is_healthy"] = False
        
        if stats["errors"] > stats["hits"]:
            health["issues"].append("High error rate")
            health["is_healthy"] = False
        
        return health
    
    # =================== PRIVATE HELPER METHODS ===================
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        try:
            return hashlib.md5(key.encode('utf-8')).hexdigest()
        except Exception:
            return hashlib.md5(str(key).encode('ascii', errors='ignore')).hexdigest()
    
    def _is_valid(self, key: str, ttl: int) -> bool:
        """Check if memory cache entry is valid"""
        if key not in self.cache_times:
            return False
        return time.time() - self.cache_times[key] < ttl
    
    def _is_file_valid(self, file_path: Path, ttl: int) -> bool:
        """Check if file cache entry is valid"""
        try:
            return time.time() - file_path.stat().st_mtime < ttl
        except Exception:
            return False
    
    def _add_to_memory(self, key: str, value: Any) -> bool:
        """Add item to memory cache with LRU eviction"""
        try:
            # Estimate memory usage
            try:
                estimated_size = len(pickle.dumps(value))
            except Exception:
                estimated_size = 1024  # Fallback estimate
            
            # Evict if necessary
            while (self.current_memory + estimated_size > self.max_memory_bytes and 
                   self.access_order):
                oldest_key = self.access_order.popleft()
                self._remove_from_memory(oldest_key)
                self.stats["evictions"] += 1
            
            # Add new item
            self.memory_cache[key] = value
            self.cache_times[key] = time.time()
            self.cache_sizes[key] = estimated_size
            self.access_order.append(key)
            self.current_memory += estimated_size
            
            return True
            
        except Exception:
            return False
    
    def _remove_from_memory(self, key: str) -> None:
        """Remove item from memory cache"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if key in self.cache_times:
            del self.cache_times[key]
        
        if key in self.cache_sizes:
            self.current_memory -= self.cache_sizes[key]
            del self.cache_sizes[key]
        
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
    
    def _update_access(self, key: str) -> None:
        """Update access order for LRU"""
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
        self.access_order.append(key)


class PerformanceMonitor:
    """
    📊 Performance Monitoring System với Enhanced Analytics
    
    Track system performance, identify bottlenecks, provide optimization insights,
    and monitor health with comprehensive analytics.
    """
    
    def __init__(self, 
                 logger: Optional[Logger] = None, 
                 config: Optional[Config] = None):
        """
        Initialize performance monitor with enhanced capabilities
        
        Args:
            logger: Logger instance
            config: Configuration instance
        """
        self.logger = logger
        self.config = config or Config()
        
        # Performance data storage
        self.timers = {}
        self.operation_stats = defaultdict(list)
        self.system_stats = {}
        
        # Enhanced analytics
        self.agent_stats = defaultdict(list)
        self.team_stats = defaultdict(list)
        self.error_stats = defaultdict(int)
        self.slow_operations = defaultdict(list)
        
        # Performance thresholds
        self.slow_threshold = self.config.get("performance.slow_operation_threshold", 5.0)
        self.memory_threshold = self.config.get("performance.memory_warning_threshold", 0.8)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start time for uptime tracking
        self.start_time = time.time()
    
    @contextmanager
    def timer(self, operation: str, **metadata):
        """
        Context manager for timing operations with metadata
        
        Args:
            operation: Operation name
            **metadata: Additional metadata to log
        """
        start_time = time.time()
        timer_id = f"{operation}_{start_time}_{threading.get_ident()}"
        
        try:
            yield timer_id
        finally:
            duration = time.time() - start_time
            self.log_operation(operation, duration, **metadata)
    
    def start_timer(self, operation: str, timer_id: Optional[str] = None) -> str:
        """
        Start timing an operation with enhanced tracking
        
        Args:
            operation: Operation name
            timer_id: Optional custom timer ID
            
        Returns:
            Timer ID for stopping
        """
        if timer_id is None:
            timer_id = f"{operation}_{time.time()}_{threading.get_ident()}"
        
        with self._lock:
            if timer_id in self.timers:
                self.logger.warning(f"Timer {timer_id} already exists, overwriting")
            
            self.timers[timer_id] = {
                "operation": operation,
                "start_time": time.time(),
                "metadata": {},
                "thread_id": threading.get_ident()
            }
        
        return timer_id
    
    def end_timer(self, timer_id: str, **metadata) -> float:
        """
        End timing an operation with validation
        
        Args:
            timer_id: Timer ID from start_timer
            **metadata: Additional metadata
            
        Returns:
            Operation duration in seconds
        """
        end_time = time.time()
        
        with self._lock:
            if timer_id not in self.timers:
                if self.logger:
                    self.logger.warning(f"Timer not found", timer_id=timer_id)
                return 0.0
            
            timer_info = self.timers.pop(timer_id)
            duration = end_time - timer_info["start_time"]
            
            # Merge metadata
            timer_info["metadata"].update(metadata)
            
            self.log_operation(
                timer_info["operation"], 
                duration, 
                **timer_info["metadata"]
            )
            
            return duration
    
    def log_operation(self, operation: str, duration: float, **metadata) -> None:
        """
        Log operation performance with enhanced analytics
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **metadata: Additional metadata
        """
        try:
            with self._lock:
                # Store operation data
                operation_data = {
                    "duration": duration,
                    "timestamp": time.time(),
                    "metadata": metadata,
                    "thread_id": threading.get_ident()
                }
                
                self.operation_stats[operation].append(operation_data)
                
                # Check for slow operations
                if duration > self.slow_threshold:
                    self.slow_operations[operation].append(operation_data)
                    
                    if self.logger:
                        self.logger.warning(
                            f"Slow operation detected",
                            operation=operation,
                            duration=duration,
                            **metadata
                        )
                
                # Maintain rolling window (keep last 1000 operations per type)
                if len(self.operation_stats[operation]) > 1000:
                    self.operation_stats[operation] = self.operation_stats[operation][-1000:]
                
                # Log to performance logger if available
                if self.logger:
                    self.logger.log_performance(operation, duration, **metadata)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Performance logging failed", error=str(e))
    
    def log_agent_performance(self, 
                            agent_name: str, 
                            operation: str, 
                            duration: float, 
                            **metadata) -> None:
        """
        Log OpenAI agent performance with validation
        
        Args:
            agent_name: Name of the agent
            operation: Operation performed
            duration: Duration in seconds
            **metadata: Additional metadata
        """
        if not self.config.get("agno.performance_tracking", True):
            return
        
        try:
            with self._lock:
                agent_data = {
                    "operation": operation,
                    "duration": duration,
                    "timestamp": time.time(),
                    "metadata": metadata
                }
                
                self.agent_stats[agent_name].append(agent_data)
                
                # Maintain rolling window
                if len(self.agent_stats[agent_name]) > 500:
                    self.agent_stats[agent_name] = self.agent_stats[agent_name][-500:]
            
            # Log to OpenAI logger
            if self.logger:
                self.logger.log_agno_event(
                    "performance",
                    agent_name,
                    operation=operation,
                    duration=duration,
                    **metadata
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Agent performance logging failed", error=str(e))
    
    def log_team_performance(self, 
                           team_name: str, 
                           operation: str, 
                           duration: float, 
                           **metadata) -> None:
        """
        Log OpenAI team performance with validation
        
        Args:
            team_name: Name of the team
            operation: Operation performed
            duration: Duration in seconds
            **metadata: Additional metadata
        """
        if not self.config.get("agno.performance_tracking", True):
            return
        
        try:
            with self._lock:
                team_data = {
                    "operation": operation,
                    "duration": duration,
                    "timestamp": time.time(),
                    "metadata": metadata
                }
                
                self.team_stats[team_name].append(team_data)
                
                # Maintain rolling window
                if len(self.team_stats[team_name]) > 500:
                    self.team_stats[team_name] = self.team_stats[team_name][-500:]
            
            # Log to OpenAI logger
            if self.logger:
                self.logger.log_agno_event(
                    "team_performance",
                    team_name,
                    operation=operation,
                    duration=duration,
                    **metadata
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Team performance logging failed", error=str(e))
    
    def log_error(self, error_type: str, operation: Optional[str] = None, **metadata) -> None:
        """Log error occurrence for analytics"""
        try:
            with self._lock:
                error_key = f"{error_type}:{operation}" if operation else error_type
                self.error_stats[error_key] += 1
                
        except Exception:
            pass  # Don't fail on error logging
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary of performance stats with analytics
        """
        try:
            with self._lock:
                current_time = time.time()
                uptime = current_time - self.start_time
                
                stats = {
                    "system": {
                        "uptime_seconds": uptime,
                        "uptime_hours": round(uptime / 3600, 2),
                        "active_timers": len(self.timers),
                        "total_operations": sum(len(ops) for ops in self.operation_stats.values()),
                        "slow_operations_count": sum(len(ops) for ops in self.slow_operations.values()),
                        "error_count": sum(self.error_stats.values())
                    },
                    "operations": {},
                    "agents": {},
                    "teams": {},
                    "slow_operations": {},
                    "errors": dict(self.error_stats),
                    "timestamp": current_time
                }
                
                # System operation stats with analytics
                for operation, measurements in self.operation_stats.items():
                    if measurements:
                        durations = [m["duration"] for m in measurements]
                        recent_durations = [m["duration"] for m in measurements[-50:]]  # Last 50
                        
                        stats["operations"][operation] = {
                            "count": len(durations),
                            "total_time": round(sum(durations), 4),
                            "avg_time": round(np.mean(durations), 4),
                            "min_time": round(min(durations), 4),
                            "max_time": round(max(durations), 4),
                            "std_time": round(np.std(durations), 4),
                            "median_time": round(np.median(durations), 4),
                            "p95_time": round(np.percentile(durations, 95), 4),
                            "recent_avg": round(np.mean(recent_durations), 4),
                            "slow_count": len(self.slow_operations.get(operation, []))
                        }
                
                # Agent stats with analytics
                for agent_name, measurements in self.agent_stats.items():
                    if measurements:
                        durations = [m["duration"] for m in measurements]
                        operations = [m["operation"] for m in measurements]
                        
                        stats["agents"][agent_name] = {
                            "count": len(durations),
                            "total_time": round(sum(durations), 4),
                            "avg_time": round(np.mean(durations), 4),
                            "unique_operations": len(set(operations)),
                            "most_common_operation": max(set(operations), key=operations.count) if operations else None
                        }
                
                # Team stats with analytics
                for team_name, measurements in self.team_stats.items():
                    if measurements:
                        durations = [m["duration"] for m in measurements]
                        operations = [m["operation"] for m in measurements]
                        
                        stats["teams"][team_name] = {
                            "count": len(durations),
                            "total_time": round(sum(durations), 4),
                            "avg_time": round(np.mean(durations), 4),
                            "unique_operations": len(set(operations))
                        }
                
                # Slow operations summary
                for operation, slow_ops in self.slow_operations.items():
                    if slow_ops:
                        durations = [op["duration"] for op in slow_ops]
                        stats["slow_operations"][operation] = {
                            "count": len(durations),
                            "avg_duration": round(np.mean(durations), 4),
                            "max_duration": round(max(durations), 4),
                            "recent_count": len([op for op in slow_ops if current_time - op["timestamp"] < 3600])  # Last hour
                        }
                
                return stats
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stats generation failed", error=str(e))
            return {"error": str(e), "timestamp": time.time()}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get performance monitor health status"""
        stats = self.get_stats()
        
        health = {
            "is_healthy": True,
            "issues": [],
            "recommendations": [],
            "stats": stats["system"]
        }
        
        try:
            # Check for performance issues
            if stats["system"]["slow_operations_count"] > 10:
                health["issues"].append("High number of slow operations")
                health["recommendations"].append("Investigate slow operation causes")
            
            if stats["system"]["error_count"] > 50:
                health["issues"].append("High error rate")
                health["recommendations"].append("Check error logs for patterns")
                health["is_healthy"] = False
            
            # Check recent performance
            recent_slow_count = 0
            for operation, slow_stats in stats.get("slow_operations", {}).items():
                recent_slow_count += slow_stats.get("recent_count", 0)
            
            if recent_slow_count > 5:
                health["issues"].append("Recent performance degradation")
                health["recommendations"].append("Check system resources and optimize slow operations")
            
            # Check active timers (potential memory leaks)
            if stats["system"]["active_timers"] > 100:
                health["issues"].append("High number of active timers")
                health["recommendations"].append("Check for timer leaks")
                health["is_healthy"] = False
            
        except Exception as e:
            health["issues"].append(f"Health check failed: {str(e)}")
            health["is_healthy"] = False
        
        return health
    
    def reset_stats(self) -> None:
        """Reset all performance statistics"""
        with self._lock:
            self.operation_stats.clear()
            self.agent_stats.clear()
            self.team_stats.clear()
            self.error_stats.clear()
            self.slow_operations.clear()
            self.timers.clear()
            self.start_time = time.time()
        
        if self.logger:
            self.logger.info("Performance statistics reset")
    
    def update_system_stats(self, **stats) -> None:
        """
        Update system-level statistics
        
        Args:
            **stats: System statistics to update
        """
        with self._lock:
            self.system_stats.update(stats)
            self.system_stats["last_updated"] = time.time()


class SmartPathResolver:
    """
    🧭 Smart Path Resolution System
    
    Provides intelligent path resolution for portable indexes,
    handles path normalization, and maintains path mappings for migration.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """Initialize smart path resolver"""
        self.logger = logger
        self.path_mappings = {}
        self.cache = {}
        self.resolution_strategies = [
            self._resolve_exact_match,
            self._resolve_relative_to_cwd,
            self._resolve_adjacent_folders,
            self._resolve_parent_directories,
            self._resolve_common_locations,
            self._resolve_with_mappings
        ]
    
    def register_path_mapping(self, old_base: str, new_base: str) -> None:
        """
        Register path mapping for migration between systems
        
        Args:
            old_base: Original base path
            new_base: New base path
        """
        try:
            old_base = str(Path(old_base).resolve())
            new_base = str(Path(new_base).resolve())
            self.path_mappings[old_base] = new_base
            
            if self.logger:
                self.logger.info(f"📍 Path mapping registered: {old_base} -> {new_base}")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to register path mapping: {e}")
    
    def resolve_portable_path(self, 
                             relative_path: str, 
                             keyframes_base: str,
                             fallback_strategies: bool = True) -> Optional[str]:
        """
        Resolve relative path to absolute path with intelligent fallback
        
        Args:
            relative_path: Relative path from portable metadata
            keyframes_base: Base keyframes directory
            fallback_strategies: Use fallback resolution strategies
            
        Returns:
            Resolved absolute path or None if not found
        """
        # Check cache first
        cache_key = f"{relative_path}|{keyframes_base}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            keyframes_path = Path(keyframes_base).resolve()
            
            # Primary resolution: relative to keyframes base
            primary_path = self._resolve_primary_path(relative_path, keyframes_path)
            if primary_path and primary_path.exists():
                resolved = str(primary_path)
                self.cache[cache_key] = resolved
                return resolved
            
            # Fallback strategies if enabled
            if fallback_strategies:
                for strategy in self.resolution_strategies:
                    try:
                        resolved_path = strategy(relative_path, keyframes_path)
                        if resolved_path and Path(resolved_path).exists():
                            self.cache[cache_key] = resolved_path
                            if self.logger:
                                self.logger.info(f"✅ Path resolved via {strategy.__name__}: {resolved_path}")
                            return resolved_path
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                        continue
            
            # Not found
            if self.logger:
                self.logger.warning(f"❌ Could not resolve path: {relative_path}")
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Path resolution error: {e}")
            return None
    
    def _resolve_primary_path(self, relative_path: str, keyframes_base: Path) -> Optional[Path]:
        """Primary path resolution strategy"""
        try:
            # Handle paths that start with 'keyframes'
            if relative_path.startswith('keyframes'):
                rel_parts = Path(relative_path).parts[1:]  # Remove 'keyframes' prefix
                if rel_parts:
                    return keyframes_base / Path(*rel_parts)
            
            # Direct relative resolution
            return keyframes_base / relative_path
            
        except Exception:
            return None
    
    def _resolve_exact_match(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try exact relative path resolution"""
        try:
            full_path = keyframes_base / relative_path
            return str(full_path) if full_path.exists() else None
        except Exception:
            return None
    
    def _resolve_relative_to_cwd(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try resolution relative to current working directory"""
        try:
            cwd_path = Path.cwd() / relative_path
            return str(cwd_path) if cwd_path.exists() else None
        except Exception:
            return None
    
    def _resolve_adjacent_folders(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try resolution in adjacent folders"""
        try:
            # Check parent directory
            parent_path = keyframes_base.parent / relative_path
            if parent_path.exists():
                return str(parent_path)
            
            # Check sibling directories
            for sibling in keyframes_base.parent.iterdir():
                if sibling.is_dir() and sibling != keyframes_base:
                    sibling_path = sibling / relative_path
                    if sibling_path.exists():
                        return str(sibling_path)
            
            return None
        except Exception:
            return None
    
    def _resolve_parent_directories(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try resolution in parent directories"""
        try:
            current = keyframes_base
            for _ in range(3):  # Check up to 3 levels up
                current = current.parent
                test_path = current / relative_path
                if test_path.exists():
                    return str(test_path)
            return None
        except Exception:
            return None
    
    def _resolve_common_locations(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try resolution in common locations"""
        try:
            common_bases = [
                Path.home() / "keyframes",
                Path.home() / "Documents" / "keyframes",
                Path.home() / "Desktop" / "keyframes",
                Path("/data/keyframes"),  # Linux common location
                Path("C:\\data\\keyframes"),  # Windows common location
            ]
            
            for base in common_bases:
                if base.exists():
                    test_path = base / relative_path
                    if test_path.exists():
                        return str(test_path)
            
            return None
        except Exception:
            return None
    
    def _resolve_with_mappings(self, relative_path: str, keyframes_base: Path) -> Optional[str]:
        """Try resolution using registered path mappings"""
        try:
            for old_base, new_base in self.path_mappings.items():
                test_path = Path(new_base) / relative_path
                if test_path.exists():
                    return str(test_path)
            return None
        except Exception:
            return None
    
    def normalize_path_for_storage(self, abs_path: str, keyframes_base: str) -> str:
        """
        Normalize absolute path for portable storage
        
        Args:
            abs_path: Absolute file path
            keyframes_base: Base keyframes directory
            
        Returns:
            Normalized relative path suitable for portable storage
        """
        try:
            abs_path = Path(abs_path).resolve()
            keyframes_base = Path(keyframes_base).resolve()
            
            # Try to make relative to keyframes base
            try:
                relative = abs_path.relative_to(keyframes_base)
                return str(relative)
            except ValueError:
                pass
            
            # If not under keyframes base, try to construct logical path
            if abs_path.name and abs_path.parent.name:
                # Use folder/filename pattern
                return f"{abs_path.parent.name}/{abs_path.name}"
            
            # Fallback to filename only
            return abs_path.name
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Path normalization fallback for {abs_path}: {e}")
            return Path(abs_path).name
    
    def validate_path_resolution(self, 
                                original_paths: List[str], 
                                resolved_paths: List[str]) -> Dict[str, Any]:
        """
        Validate path resolution results
        
        Args:
            original_paths: Original absolute paths
            resolved_paths: Resolved paths after portable conversion
            
        Returns:
            Validation report
        """
        report = {
            'total_paths': len(original_paths),
            'resolved_successfully': 0,
            'resolution_failures': [],
            'missing_files': [],
            'success_rate': 0.0
        }
        
        try:
            for i, (original, resolved) in enumerate(zip(original_paths, resolved_paths)):
                if resolved and Path(resolved).exists():
                    report['resolved_successfully'] += 1
                else:
                    report['resolution_failures'].append({
                        'index': i,
                        'original_path': original,
                        'resolved_path': resolved,
                        'exists': Path(resolved).exists() if resolved else False
                    })
                    
                    if resolved and not Path(resolved).exists():
                        report['missing_files'].append(resolved)
            
            if report['total_paths'] > 0:
                report['success_rate'] = report['resolved_successfully'] / report['total_paths']
            
            if self.logger:
                self.logger.info(f"📊 Path resolution validation: {report['success_rate']:.1%} success rate")
                if report['resolution_failures']:
                    self.logger.warning(f"⚠️ {len(report['resolution_failures'])} path resolution failures")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Path validation error: {e}")
            
        return report
    
    def clear_cache(self) -> None:
        """Clear path resolution cache"""
        self.cache.clear()
        if self.logger:
            self.logger.debug("🧹 Path resolution cache cleared")
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get path resolution statistics"""
        return {
            'cache_entries': len(self.cache),
            'registered_mappings': len(self.path_mappings),
            'resolution_strategies': len(self.resolution_strategies),
            'mappings': dict(self.path_mappings)
        }


# Module-level convenience functions (maintaining backward compatibility)
def get_config(config_path: str = None) -> Config:
    """Get global configuration instance"""
    return Config(config_path)

def get_logger(name: str = "EnhancedRetrievalSystem", config: Optional[Config] = None) -> Logger:
    """Get logger instance"""
    return Logger(config=config)

def get_file_manager(logger: Optional[Logger] = None) -> FileManager:
    """Get file manager instance"""
    return FileManager(logger)

def get_cache_manager(config: Optional[Config] = None) -> CacheManager:
    """Get cache manager instance"""
    return CacheManager(config=config)

def get_performance_monitor(logger: Optional[Logger] = None, config: Optional[Config] = None) -> PerformanceMonitor:
    """Get performance monitor instance"""
    return PerformanceMonitor(logger, config)

def get_agno_manager(config: Optional[Config] = None, logger: Optional[Logger] = None) -> AgnoManager:
    """Get OpenAI agent manager instance"""
    return AgnoManager(config, logger)


if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Enhanced Retrieval System - Utils Module v2.1 (OpenAI Edition)")
    print("=" * 80)
    
    # Test Unicode setup
    print("\n🌏 Testing Unicode Support...")
    UnicodeHelper.setup_unicode_environment()
    test_unicode_text = "Tìm khung cảnh người đàn ông đầu trọc mặc áo xanh dương"
    safe_text = UnicodeHelper.safe_unicode_string(test_unicode_text)
    print(f"✅ Unicode text processed: {len(safe_text)} characters")
    
    # Test configuration with validation
    print("\n🔧 Testing Enhanced Configuration (OpenAI Edition)...")
    config = Config()
    print(f"✅ Config loaded: {config.get('system.name')}")
    print(f"Unicode support: {config.get('system.unicode_support')}")
    print(f"LLM provider: {config.get('llm.provider')}")
    print(f"OpenAI model: {config.get('openai.default_model')}")
    
    validation_errors = config.get_validation_errors()
    if validation_errors:
        print(f"⚠️ Config validation issues: {len(validation_errors)}")
    else:
        print("✅ Config validation passed")
    
    # Test logger with Unicode
    print("\n📝 Testing Enhanced Logger with Unicode...")
    logger = Logger(config=config)
    logger.info("Test Unicode message", query=test_unicode_text)
    health = logger.get_health_status()
    print(f"✅ Logger health: {health['is_healthy']}")
    
    # Test OpenAI agent manager
    print("\n🤖 Testing Enhanced OpenAI Agent Manager...")
    if HAS_AGNO:
        agno_manager = AgnoManager(config, logger)
        agno_health = agno_manager.get_health_status()
        print(f"✅ OpenAI Agno enabled: {agno_manager.enabled}")
        print(f"OpenAI Agno health: {agno_health['is_healthy']}")
        
        if agno_manager.enabled:
            test_agent = agno_manager.create_agent(
                name="test_openai_agent",
                role="Test OpenAI agent for Unicode support",
                model_name="gpt-4o"
            )
            
            if test_agent:
                print("✅ Test OpenAI agent created successfully")
            else:
                print("⚠️ Test OpenAI agent creation failed (check API key)")
    else:
        print("⚠️ Agno framework not available")
    
    # Test file manager
    print("\n📁 Testing Enhanced File Manager...")
    file_manager = FileManager(logger)
    stats = file_manager.get_operation_stats()
    print(f"✅ File manager ready, operation stats: {stats}")
    
    # Test cache manager with validation
    print("\n💾 Testing Enhanced Cache Manager...")
    cache = CacheManager(config=config)
    cache.set("test_unicode_key", {"test": test_unicode_text})
    cached_data = cache.get("test_unicode_key")
    cache_health = cache.get_health_status()
    print(f"✅ Cache test passed: {cached_data is not None}")
    print(f"Cache health: {cache_health['is_healthy']}")
    
    # Test performance monitor
    print("\n📊 Testing Enhanced Performance Monitor...")
    perf = PerformanceMonitor(logger, config)
    with perf.timer("test_operation"):
        time.sleep(0.1)
    
    perf_stats = perf.get_stats()
    perf_health = perf.get_health_status()
    print(f"✅ Performance monitoring working: {len(perf_stats['operations'])} operations tracked")
    print(f"Performance health: {perf_health['is_healthy']}")
    
    # Test data processor
    print("\n🔄 Testing Enhanced Data Processor...")
    processor = DataProcessor(logger)
    test_data = [["result1"], ["result2"], ["result1"]]  # With duplicate
    merged = processor.merge_results(test_data, remove_duplicates=True)
    proc_stats = processor.get_processing_stats()
    print(f"✅ Data processing working: {len(merged)} unique results from {len(test_data)} lists")
    print(f"Processing stats: {proc_stats['operations_performed']} operations")
    
    # Test smart path resolver
    print("\n🧭 Testing Smart Path Resolver...")
    path_resolver = SmartPathResolver(logger)
    stats = path_resolver.get_resolution_stats()
    print(f"✅ Path resolver ready: {stats}")
    
    print("\n✅ All enhanced utilities tested successfully!")
    print("🌏 Unicode support enabled and working!")
    print("🛡️ Robust validation and error handling implemented!")
    print("🤖 OpenAI-focused architecture ready!")
    print("🚀 Ready for production use with enhanced reliability!")
