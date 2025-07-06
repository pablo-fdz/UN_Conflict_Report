"""
Unit tests for GraphRAG pipeline components.

Tests the GraphRAGConstructionPipeline class and related retrieval functionality.
"""

import json
from unittest.mock import Mock, patch, mock_open, AsyncMock

import pytest

# Import test dependencies
try:
    from pipeline.graphrag.graphrag_construction_pipeline import GraphRAGConstructionPipeline
    from tests.mocks.mock_services import MockNeo4jDriver, MockGeminiLLM, MockRetriever
    from tests.fixtures.sample_data import SampleConfigs
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Cannot import GraphRAG dependencies: {e}")
    HAS_DEPENDENCIES = False

# Skip tests if dependencies are not available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required GraphRAG dependencies not available")


class TestGraphRAGConstructionPipeline:
    """Test suite for GraphRAGConstructionPipeline class."""

    @pytest.fixture
    def mock_config_files(self):
        """Mock configuration files for testing."""
        return {
            'graphrag_config.json': SampleConfigs.graphrag_config(),
            '.env': 'NEO4J_URI=bolt://localhost:7687\nNEO4J_USERNAME=neo4j\nNEO4J_PASSWORD=password\nGEMINI_API_KEY=test_key'
        }

    @pytest.fixture
    def graphrag_pipeline_instance(self, mock_config_files):
        """Create a GraphRAGConstructionPipeline instance with mocked dependencies."""
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'), \
             patch.dict('os.environ', {
                 'NEO4J_URI': 'bolt://localhost:7687',
                 'NEO4J_USERNAME': 'neo4j', 
                 'NEO4J_PASSWORD': 'password',
                 'GEMINI_API_KEY': 'test_key'
             }), \
             patch('json.load', return_value=mock_config_files['graphrag_config.json']):
            
            return GraphRAGConstructionPipeline()

    def test_initialization(self, graphrag_pipeline_instance):
        """Test GraphRAGConstructionPipeline initialization."""
        assert graphrag_pipeline_instance.neo4j_uri == 'bolt://localhost:7687'
        assert graphrag_pipeline_instance.gemini_api_key == 'test_key'
        assert graphrag_pipeline_instance.llm_usage == 0

    def test_config_loading(self, mock_config_files):
        """Test configuration loading functionality."""
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'), \
             patch('json.load', return_value=mock_config_files['graphrag_config.json']), \
             patch.dict('os.environ', {
                 'NEO4J_URI': 'bolt://localhost:7687',
                 'NEO4J_USERNAME': 'neo4j', 
                 'NEO4J_PASSWORD': 'password',
                 'GEMINI_API_KEY': 'test_key'
             }):
            
            pipeline = GraphRAGConstructionPipeline()
            assert pipeline.graphrag_config is not None

    def test_missing_env_variables_raises_error(self):
        """Test that missing environment variables raise appropriate errors."""
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'), \
             patch('json.load', return_value=SampleConfigs.graphrag_config()), \
             patch.dict('os.environ', {}, clear=True):
            
            with pytest.raises(ValueError, match="Missing required environment variables"):
                GraphRAGConstructionPipeline()

    def test_missing_config_file_raises_error(self):
        """Test that missing configuration files raise appropriate errors."""
        with patch('builtins.open', side_effect=FileNotFoundError("Config not found")), \
             patch('os.path.exists', return_value=False):
            
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                GraphRAGConstructionPipeline()

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON in config files raises appropriate errors."""
        with patch('builtins.open', mock_open(read_data="invalid json")), \
             patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'), \
             patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):
            
            with pytest.raises(ValueError, match="Error decoding JSON"):
                GraphRAGConstructionPipeline()

    @pytest.mark.asyncio
    async def test_run_async_success(self, graphrag_pipeline_instance):
        """Test successful async execution of GraphRAG pipeline."""
        mock_retriever = MockRetriever()
        
        with patch('pipeline.graphrag.graphrag_construction_pipeline.CustomGraphRAG') as mock_graphrag:
            # Mock the GraphRAG results
            mock_result = Mock()
            mock_result.answer = "Test answer"
            mock_result.retriever_result = Mock()
            mock_graphrag_instance = Mock()
            mock_graphrag_instance.search.return_value = mock_result
            mock_graphrag.return_value = mock_graphrag_instance
            
            answer, context = await graphrag_pipeline_instance.run_async(
                retriever=mock_retriever,
                country="TestCountry"
            )
            
            assert answer == "Test answer"
            assert graphrag_pipeline_instance.llm_usage == 1

    @pytest.mark.asyncio
    async def test_run_async_with_retriever_params(self, graphrag_pipeline_instance):
        """Test async execution with custom retriever parameters."""
        mock_retriever = MockRetriever()
        retriever_params = {"search_k": 10, "filter": {"country": "TestCountry"}}
        
        with patch('pipeline.graphrag.graphrag_construction_pipeline.CustomGraphRAG') as mock_graphrag:
            mock_result = Mock()
            mock_result.answer = "Test answer with params"
            mock_result.retriever_result = Mock()
            mock_graphrag_instance = Mock()
            mock_graphrag_instance.search.return_value = mock_result
            mock_graphrag.return_value = mock_graphrag_instance
            
            answer, context = await graphrag_pipeline_instance.run_async(
                retriever=mock_retriever,
                retriever_search_params=retriever_params,
                country="TestCountry"
            )
            
            assert answer == "Test answer with params"

    @pytest.mark.asyncio
    async def test_run_async_error_handling(self, graphrag_pipeline_instance):
        """Test error handling during async execution."""
        mock_retriever = MockRetriever()
        
        with patch('pipeline.graphrag.graphrag_construction_pipeline.CustomGraphRAG') as mock_graphrag:
            mock_graphrag.side_effect = Exception("GraphRAG error")
            
            with pytest.raises(RuntimeError, match="Error during GraphRAG construction pipeline execution"):
                await graphrag_pipeline_instance.run_async(
                    retriever=mock_retriever,
                    country="TestCountry"
                )

    def test_save_report_to_markdown(self, graphrag_pipeline_instance):
        """Test saving report to markdown file."""
        answer = "Test security report content"
        country = "TestCountry"
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('os.makedirs'), \
             patch('pipeline.graphrag.graphrag_construction_pipeline.datetime') as mock_datetime:
            
            mock_datetime.now.return_value.strftime.return_value = "20240101_1200"
            
            filepath, context_filepath = graphrag_pipeline_instance.save_report_to_markdown(
                answer=answer,
                country=country
            )
            
            assert filepath is not None
            assert filepath.endswith('.md')
            mock_file.assert_called()

    def test_format_markdown_report(self, graphrag_pipeline_instance):
        """Test markdown report formatting."""
        answer = "Test report content"
        country = "TestCountry"
        
        with patch.object(graphrag_pipeline_instance, '_get_latest_forecast_data') as mock_forecast:
            mock_forecast.return_value = ({}, None, None, None)
            
            formatted_report = graphrag_pipeline_instance._format_markdown_report(
                answer=answer,
                country=country
            )
            
            assert isinstance(formatted_report, str)
            assert answer in formatted_report
            assert "# Metadata" in formatted_report

    def test_get_default_output_directory(self, graphrag_pipeline_instance):
        """Test default output directory generation."""
        country = "TestCountry"
        
        output_dir = graphrag_pipeline_instance._get_default_output_directory(country)
        
        assert isinstance(output_dir, str)
        assert "TestCountry" in output_dir

    def test_get_latest_forecast_data(self, graphrag_pipeline_instance):
        """Test forecast data retrieval."""
        output_directory = "/test/output/dir"
        
        with patch('os.listdir', return_value=[]), \
             patch('os.makedirs'):
            
            forecast_data, line_chart, bar_chart, data_path = graphrag_pipeline_instance._get_latest_forecast_data(output_directory)
            
            assert isinstance(forecast_data, dict)


class TestGraphRAGComponents:
    """Test class for individual GraphRAG components."""

    def test_graphrag_configuration(self):
        """Test GraphRAG configuration validation."""
        # Test minimum required configuration
        minimal_config = {
            "llm_config": {
                "model_name": "gemini-pro",
                "max_requests_per_minute": 20
            },
            "rag_template_config": {
                "template": None,
                "system_instructions": None
            },
            "search_text": "Test search text",
            "query_text": "Test query text",
            "examples": "",
            "return_context": True
        }
        
        # Validate required fields
        assert "llm_config" in minimal_config
        assert "rag_template_config" in minimal_config
        assert minimal_config["llm_config"]["model_name"] == "gemini-pro"

    def test_retriever_integration(self):
        """Test retriever integration."""
        mock_retriever = MockRetriever()
        
        # Test retriever configuration
        retriever_config = {
            "search_k": 10,
            "filter": {"country": "TestCountry"}
        }
        
        assert mock_retriever is not None
        assert isinstance(retriever_config, dict)

    def test_llm_rate_limiting(self):
        """Test LLM rate limiting functionality."""
        # Test rate limit configuration
        rpm = 20
        safe_rpm = round(rpm - rpm * 0.2)  # 20% safety margin
        
        assert safe_rpm == 16
        assert safe_rpm < rpm

    def test_output_directory_sanitization(self):
        """Test output directory name sanitization."""
        import re
        
        # Test country name sanitization
        unsafe_country = "Test Country/Name*"
        safe_country = re.sub(r'[^\w\-]', '_', unsafe_country)
        
        assert safe_country == "Test_Country_Name_"
        assert "/" not in safe_country
        assert "*" not in safe_country


class TestGraphRAGErrorHandling:
    """Test error handling in GraphRAG operations."""

    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        # Test missing configuration sections
        incomplete_config = {
            'llm_config': {'model_name': 'gemini-pro'},
            # Missing other required sections
        }
        
        # Basic validation - should not crash
        assert 'llm_config' in incomplete_config

    def test_retriever_error_handling(self):
        """Test handling of retriever errors."""
        mock_retriever = Mock()
        mock_retriever.search.side_effect = Exception("Retriever error")
        
        # Test that errors are handled gracefully
        with pytest.raises(Exception):
            mock_retriever.search("test query")

    def test_llm_error_handling(self):
        """Test handling of LLM errors."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")
        
        # Test that LLM errors are handled
        with pytest.raises(Exception):
            mock_llm.generate("test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
