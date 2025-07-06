"""
Mock objects for testing pipeline components.

This module provides mock implementations of external services and dependencies
to enable isolated unit testing.
"""

from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock
import polars as pl
import neo4j
from tests.fixtures.sample_data import (
    SAMPLE_ACLED_DATA, SAMPLE_FACTIVA_DATA, SAMPLE_GOOGLE_NEWS_DATA
)

class MockACLEDAPI:
    """Mock ACLED API for testing data ingestion."""
    
    def __init__(self, return_data: Optional[List[Dict]] = None):
        self.return_data = return_data or SAMPLE_ACLED_DATA
        self.call_count = 0
        
    def get_data(self, country: str, start_date: str, end_date: str, **kwargs) -> pl.DataFrame:
        """Mock API call that returns sample data."""
        self.call_count += 1
        self.last_params = {
            'country': country,
            'start_date': start_date, 
            'end_date': end_date,
            **kwargs
        }
        return pl.DataFrame(self.return_data)
    
    def simulate_api_error(self):
        """Simulate API error for testing error handling."""
        raise ConnectionError("ACLED API unavailable")

class MockFactivaAPI:
    """Mock Factiva API for testing data ingestion."""
    
    def __init__(self, return_data: Optional[List[Dict]] = None):
        self.return_data = return_data or SAMPLE_FACTIVA_DATA
        self.call_count = 0
        
    def search(self, query: str, max_results: int = 100, **kwargs) -> pl.DataFrame:
        """Mock search that returns sample data."""
        self.call_count += 1
        self.last_query = query
        self.last_max_results = max_results
        return pl.DataFrame(self.return_data)

class MockGoogleNewsAPI:
    """Mock Google News API for testing data ingestion."""
    
    def __init__(self, return_data: Optional[List[Dict]] = None):
        self.return_data = return_data or SAMPLE_GOOGLE_NEWS_DATA
        self.call_count = 0
        
    def search(self, query: str, max_results: int = 100, **kwargs) -> pl.DataFrame:
        """Mock search that returns sample data."""
        self.call_count += 1
        self.last_query = query
        return pl.DataFrame(self.return_data)

class MockNeo4jDriver(neo4j.Driver):
    """Mock Neo4j driver for testing database operations."""
    
    def __init__(self):
        # Don't call super().__init__() to avoid actual connection
        self.sessions = []
        self.is_connected = True
        self.query_history = []
        
        # Internal attributes that the real Neo4j driver has
        self._query_bookmark_manager = Mock()
        self._query_timeout = 30
        self._encrypted = False
        self._auth = None
        self._user_agent = "mock-driver/1.0"
        self._trust = Mock()
        self._resolver = Mock()
        self._connection_pool = Mock()
        
        # Missing attribute that Neo4j 5.x requires
        self._default_workspace_config = Mock()
        
    def session(self, **kwargs):
        """Return a mock session."""
        session = MockNeo4jSession()
        self.sessions.append(session)
        return session
        
    def execute_query(self, query: str, parameters=None, **kwargs):
        """Mock execute_query method that Neo4j GraphRAG uses."""
        # Log the query for debugging
        self.query_history.append({
            'query': query,
            'parameters': parameters or {},
            'kwargs': kwargs
        })
        
        # Return a mock result with the expected structure
        from unittest.mock import Mock
        mock_result = Mock()
        mock_result.records = []
        mock_result.summary = Mock()
        return mock_result
        
    async def verify_connectivity(self):
        """Mock connectivity verification."""
        if not self.is_connected:
            raise neo4j.exceptions.ServiceUnavailable("Database unavailable")
        return True
        
    def close(self):
        """Mock driver close."""
        pass

class MockNeo4jSession:
    """Mock Neo4j session for testing database operations."""
    
    def __init__(self):
        self.query_history = []
        self.is_open = True
        self.mock_results = {}
        
        # Internal attributes that the real session has
        self._connection = Mock()
        self._bookmark_manager = Mock()
        
    def run(self, query: str, parameters: Optional[Dict] = None):
        """Mock query execution."""
        self.query_history.append({
            'query': query,
            'parameters': parameters or {}
        })
        
        # Return mock results based on query type
        if "CREATE" in query.upper():
            return MockNeo4jResult([{"created": True}])
        elif "MATCH" in query.upper():
            return MockNeo4jResult(self.mock_results.get('match', []))
        elif "COUNT" in query.upper():
            return MockNeo4jResult([{"count": 10}])
        else:
            return MockNeo4jResult([])
    
    def set_mock_result(self, query_type: str, result: List[Dict]):
        """Set mock result for a specific query type."""
        self.mock_results[query_type] = result
        
    def close(self):
        """Mock session close."""
        self.is_open = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class MockNeo4jResult:
    """Mock Neo4j query result."""
    
    def __init__(self, data: List[Dict]):
        self._data = data
        self._records = [MockRecord(record) for record in data]
        
    def data(self):
        """Return mock result data."""
        return self._data
        
    def single(self):
        """Return single mock result."""
        return self._records[0] if self._records else None
        
    def records(self):
        """Return list of mock records."""
        return self._records
        
    def __iter__(self):
        """Allow iteration over records."""
        return iter(self._records)

class MockRecord:
    """Mock Neo4j record."""
    
    def __init__(self, data: Dict):
        self._data = data
        
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._data[key]
        
    def get(self, key, default=None):
        """Get value with default."""
        return self._data.get(key, default)
        
    def data(self):
        """Return record data."""
        return self._data

class MockLLMResponse:
    """Mock LLM response that matches the real LLMResponse interface."""
    
    def __init__(self, content: str, parsed: Optional[Dict] = None):
        self.content = content  # This is what the real interface expects
        self.parsed = parsed

class MockLLM:
    """Mock LLM for testing knowledge graph building and GraphRAG."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_history = []
        self.default_response = "Mock LLM response"
        
    def invoke(self, input: str, message_history=None, system_instruction=None) -> MockLLMResponse:
        """Mock invoke method that matches the real LLMInterface."""
        self.call_history.append({
            'input': input,
            'message_history': message_history,
            'system_instruction': system_instruction
        })
        
        # Return predefined response or default
        for key, response in self.responses.items():
            if key.lower() in input.lower():
                return MockLLMResponse(content=response)
        return MockLLMResponse(content=self.default_response)
    
    async def ainvoke(self, input: str, message_history=None, system_instruction=None) -> MockLLMResponse:
        """Mock async invoke method."""
        return self.invoke(input, message_history, system_instruction)
        
    # Legacy methods for backward compatibility
    def generate(self, prompt: str, **kwargs) -> str:
        """Legacy mock text generation method."""
        response = self.invoke(prompt, **kwargs)
        return response.content
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Legacy mock async text generation method."""
        response = await self.ainvoke(prompt, **kwargs)
        return response.content
    
    def set_response(self, trigger: str, response: str):
        """Set a specific response for prompts containing trigger text."""
        self.responses[trigger] = response

class MockCustomKGPipeline:
    """Mock custom KG pipeline for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    async def run(self, documents: List[Dict]) -> Dict:
        """Mock KG pipeline execution."""
        return {
            "status": "success",
            "entities_created": len(documents) * 2,
            "relationships_created": len(documents) * 3
        }


class MockGraphRAG:
    """Mock GraphRAG system for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.search = AsyncMock()
    
    async def search(self, query: str, **kwargs) -> str:
        """Mock GraphRAG search."""
        return f"Mock GraphRAG response for query: {query[:30]}..."


class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        from unittest.mock import AsyncMock
        self.search = AsyncMock()
    
    async def search(self, query: str, **kwargs) -> List:
        """Mock search operation."""
        from unittest.mock import Mock
        # Return mock search results
        return [
            Mock(content="Mock relevant content 1", metadata={"score": 0.9}),
            Mock(content="Mock relevant content 2", metadata={"score": 0.8})
        ]


class MockGeminiLLM:
    """Mock Gemini LLM for testing."""
    
    def __init__(self, model_name: str = "gemini-pro", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def ainvoke(self, prompt: str, **kwargs) -> MockLLMResponse:
        """Mock async LLM invocation."""
        return MockLLMResponse(content=f"Mock response for: {prompt[:50]}...")
    
    def invoke(self, prompt: str, **kwargs) -> MockLLMResponse:
        """Mock sync LLM invocation."""
        return MockLLMResponse(content=f"Mock response for: {prompt[:50]}...")


class MockEmbeddings:
    """Mock embeddings model for testing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimensions = 384
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Mock document embedding."""
        return [[0.1] * self.dimensions for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Mock query embedding."""
        return [0.1] * self.dimensions


class MockResolver:
    """Mock entity resolver for testing."""
    
    def __init__(self, driver, **kwargs):
        self.driver = driver
        self.kwargs = kwargs
    
    def resolve(self, entities: List[Dict]) -> List[Dict]:
        """Mock entity resolution."""
        # Simple mock: just return the same entities
        return entities

class MockRateLimiter:
    """Mock rate limiter for testing API calls."""
    
    def __init__(self, max_calls: int = 100):
        self.max_calls = max_calls
        self.current_calls = 0
        self.reset_count = 0
        
    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.current_calls < self.max_calls
    
    def increment(self):
        """Increment call count."""
        self.current_calls += 1
        
    def reset(self):
        """Reset call count."""
        self.current_calls = 0
        self.reset_count += 1

class MockConfigLoader:
    """Mock configuration loader for testing."""
    
    def __init__(self, config: Optional[Dict] = None):
        from tests.fixtures.sample_data import SAMPLE_CONFIG
        self.config = config or SAMPLE_CONFIG
        
    def load_config(self, config_type: str) -> Dict:
        """Load mock configuration."""
        return self.config.get(config_type, {})
    
    def get_env_var(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get mock environment variable."""
        mock_env = {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password',
            'ACLED_EMAIL': 'test@example.com',
            'ACLED_API_KEY': 'mock-api-key',
            'GEMINI_API_KEY': 'mock-gemini-key'
        }
        return mock_env.get(var_name, default)

class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
        
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        return path in self.files
        
    def read_text(self, path: str) -> str:
        """Read file content."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]
        
    def write_text(self, path: str, content: str):
        """Write file content."""
        self.files[path] = content
        
    def mkdir(self, path: str):
        """Create directory."""
        self.directories.add(path)
        
    def add_file(self, path: str, content: str):
        """Add a file to mock filesystem."""
        self.files[path] = content

def create_mock_pipeline_context():
    """Create a complete mock context for pipeline testing."""
    return {
        'neo4j_driver': MockNeo4jDriver(),
        'llm': MockLLM(),
        'embeddings': MockEmbeddings(),
        'retriever': MockRetriever(),
        'rate_limiter': MockRateLimiter(),
        'config_loader': MockConfigLoader(),
        'file_system': MockFileSystem()
    }
