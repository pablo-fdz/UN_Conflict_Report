"""
Configuration file for pytest.

This file handles path setup and common test configuration.
"""

import sys
import os
from pathlib import Path

# Add the graphrag_pipeline directory to the Python path for testing
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Set up environment variables for testing
os.environ.setdefault('TESTING', 'True')
os.environ.setdefault('GOOGLE_AI_API_KEY', 'test_key')
os.environ.setdefault('NEO4J_URI', 'bolt://localhost:7687')
os.environ.setdefault('NEO4J_USERNAME', 'neo4j')
os.environ.setdefault('NEO4J_PASSWORD', 'password')
