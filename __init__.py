"""
OmniRCA — ReAct Agentic Root Cause Analysis for Microservices
Phase 1: Data Layer & Tool Library
"""
from .data_layer.loader import get_loader
from .data_layer.graph_loader import get_arch_graph
from .data_layer.kv_store import get_store
from .logger import setup_logging

setup_logging()  # configure timestamped logging once at import time

__version__ = "0.1.0"
