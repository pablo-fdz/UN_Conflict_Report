import neo4j
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    HybridCypherRetriever,
    Text2CypherRetriever
)
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.types import (
    RetrieverResultItem,
    RawSearchResult,
    HybridSearchRanker
)


class RetrieverType(Enum):
    """Enum representing the available retriever types."""
    VECTOR = "vector"
    VECTOR_CYPHER = "vector_cypher"
    HYBRID = "hybrid"
    HYBRID_CYPHER = "hybrid_cypher"
    TEXT2CYPHER = "text2cypher"


class KGRetriever:
    """
    A unified interface for Neo4j GraphRAG retrievers.
    
    This class provides factory methods to create different types of retrievers
    and a unified interface to search across all of them.
    """
    
    def __init__(
        self, 
        retriever_type: RetrieverType,
        retriever: Any
    ):
        """
        Initialize a KGRetriever instance with a specific retriever type.
        
        Args:
            retriever_type: The type of retriever being used.
            retriever: The actual retriever instance.
        """
        self.retriever_type = retriever_type
        self.retriever = retriever
        
    @classmethod
    def create_vector_retriever(
        cls,
        driver: neo4j.Driver,
        index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[List[str]] = None,
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        neo4j_database: Optional[str] = None
    ) -> "KGRetriever":
        """
        Create a KGRetriever with a VectorRetriever.
        
        Args:
            driver: The Neo4j Python driver.
            index_name: Vector index name.
            embedder: Optional embedder object to embed query text.
            return_properties: Optional list of node properties to return.
            result_formatter: Optional custom function to transform a neo4j.Record to a RetrieverResultItem.
            neo4j_database: Optional name of the Neo4j database.
            
        Returns:
            A KGRetriever instance containing a VectorRetriever.
        """
        retriever = VectorRetriever(
            driver=driver,
            index_name=index_name,
            embedder=embedder,
            return_properties=return_properties,
            result_formatter=result_formatter,
            neo4j_database=neo4j_database
        )
        return cls(RetrieverType.VECTOR, retriever)
    
    @classmethod
    def create_vector_cypher_retriever(
        cls,
        driver: neo4j.Driver,
        index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        neo4j_database: Optional[str] = None
    ) -> "KGRetriever":
        """
        Create a KGRetriever with a VectorCypherRetriever.
        
        Args:
            driver: The Neo4j Python driver.
            index_name: Vector index name.
            retrieval_query: Cypher query that gets appended.
            embedder: Optional embedder object to embed query text.
            result_formatter: Optional custom function to transform a neo4j.Record to a RetrieverResultItem.
            neo4j_database: Optional name of the Neo4j database.
            
        Returns:
            A KGRetriever instance containing a VectorCypherRetriever.
        """
        retriever = VectorCypherRetriever(
            driver=driver,
            index_name=index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=result_formatter,
            neo4j_database=neo4j_database
        )
        return cls(RetrieverType.VECTOR_CYPHER, retriever)
    
    @classmethod
    def create_hybrid_retriever(
        cls,
        driver: neo4j.Driver,
        vector_index_name: str,
        fulltext_index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[List[str]] = None,
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        neo4j_database: Optional[str] = None
    ) -> "KGRetriever":
        """
        Create a KGRetriever with a HybridRetriever.
        
        Args:
            driver: The Neo4j Python driver.
            vector_index_name: Vector index name.
            fulltext_index_name: Fulltext index name.
            embedder: Optional embedder object to embed query text.
            return_properties: Optional list of node properties to return.
            result_formatter: Optional custom function to transform a neo4j.Record to a RetrieverResultItem.
            neo4j_database: Optional name of the Neo4j database.
            
        Returns:
            A KGRetriever instance containing a HybridRetriever.
        """
        retriever = HybridRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=embedder,
            return_properties=return_properties,
            result_formatter=result_formatter,
            neo4j_database=neo4j_database
        )
        return cls(RetrieverType.HYBRID, retriever)
    
    @classmethod
    def create_hybrid_cypher_retriever(
        cls,
        driver: neo4j.Driver,
        vector_index_name: str,
        fulltext_index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        neo4j_database: Optional[str] = None
    ) -> "KGRetriever":
        """
        Create a KGRetriever with a HybridCypherRetriever.
        
        Args:
            driver: The Neo4j Python driver.
            vector_index_name: Vector index name.
            fulltext_index_name: Fulltext index name.
            retrieval_query: Cypher query that gets appended.
            embedder: Optional embedder object to embed query text.
            result_formatter: Optional custom function to transform a neo4j.Record to a RetrieverResultItem.
            neo4j_database: Optional name of the Neo4j database.
            
        Returns:
            A KGRetriever instance containing a HybridCypherRetriever.
        """
        retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=result_formatter,
            neo4j_database=neo4j_database
        )
        return cls(RetrieverType.HYBRID_CYPHER, retriever)
    
    @classmethod
    def create_text2cypher_retriever(
        cls,
        driver: neo4j.Driver,
        llm: LLMInterface,
        neo4j_schema: Optional[str] = None,
        examples: Optional[List[str]] = None,
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        custom_prompt: Optional[str] = None,
        neo4j_database: Optional[str] = None
    ) -> "KGRetriever":
        """
        Create a KGRetriever with a Text2CypherRetriever.
        
        Args:
            driver: The Neo4j Python driver.
            llm: LLM object to generate the Cypher query.
            neo4j_schema: Optional Neo4j schema used to generate the Cypher query.
            examples: Optional user input/query pairs for the LLM to use as examples.
            result_formatter: Optional custom function to transform a neo4j.Record to a RetrieverResultItem.
            custom_prompt: Optional custom prompt to use instead of auto-generated prompt.
            neo4j_database: Optional name of the Neo4j database.
            
        Returns:
            A KGRetriever instance containing a Text2CypherRetriever.
        """
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=neo4j_schema,
            examples=examples,
            result_formatter=result_formatter,
            custom_prompt=custom_prompt,
            neo4j_database=neo4j_database
        )
        return cls(RetrieverType.TEXT2CYPHER, retriever)
    
    def get_search_results(self, **kwargs) -> RawSearchResult:
        """
        Get search results from the underlying retriever.
        
        This method passes the appropriate arguments to the retriever's get_search_results method
        based on the retriever type.
        
        Args:
            **kwargs: Keyword arguments to pass to the retriever's get_search_results method.
                Common parameters include:
                - query_text: The text to search for.
                - query_vector: The vector to search for (optional).
                - top_k: The number of results to return (default: 5).
                
                Vector and Hybrid retrievers also support:
                - effective_search_ratio: Controls the candidate pool size (default: 1).
                
                Vector retrievers support:
                - filters: Filters for metadata pre-filtering (optional).
                
                Cypher retrievers support:
                - query_params: Parameters for the Cypher query (optional).
                
                Hybrid retrievers support:
                - ranker: Type of ranker to order the results (default: naive).
                - alpha: Weight for the vector score when using the linear ranker (optional).
                
                Text2Cypher retrievers support:
                - prompt_params: Additional values to inject into the custom prompt (optional).
                
        Returns:
            The raw search results from the retriever.
        """
        if self.retriever_type == RetrieverType.VECTOR:
            allowed_params = ['query_vector', 'query_text', 'top_k', 'effective_search_ratio', 'filters']
            params = {k: v for k, v in kwargs.items() if k in allowed_params}
            return self.retriever.get_search_results(**params)
            
        elif self.retriever_type == RetrieverType.VECTOR_CYPHER:
            allowed_params = ['query_vector', 'query_text', 'top_k', 'effective_search_ratio', 'query_params', 'filters']
            params = {k: v for k, v in kwargs.items() if k in allowed_params}
            return self.retriever.get_search_results(**params)
            
        elif self.retriever_type == RetrieverType.HYBRID:
            allowed_params = ['query_text', 'query_vector', 'top_k', 'effective_search_ratio', 'ranker', 'alpha']
            params = {k: v for k, v in kwargs.items() if k in allowed_params}
            return self.retriever.get_search_results(**params)
            
        elif self.retriever_type == RetrieverType.HYBRID_CYPHER:
            allowed_params = ['query_text', 'query_vector', 'top_k', 'effective_search_ratio', 'query_params', 'ranker', 'alpha']
            params = {k: v for k, v in kwargs.items() if k in allowed_params}
            return self.retriever.get_search_results(**params)
            
        elif self.retriever_type == RetrieverType.TEXT2CYPHER:
            allowed_params = ['query_text', 'prompt_params']
            params = {k: v for k, v in kwargs.items() if k in allowed_params}
            return self.retriever.get_search_results(**params)
            
        raise ValueError(f"Unknown retriever type: {self.retriever_type}")
    
    def search(self, query_text: str, **kwargs) -> List[RetrieverResultItem]:
        """
        Search the knowledge graph using the underlying retriever.
        
        This is a simplified wrapper around get_search_results that formats the results.
        
        Args:
            query_text: The text to search for.
            **kwargs: Additional keyword arguments to pass to get_search_results.
                See get_search_results for details.
                
        Returns:
            A list of formatted retriever results.
        """
        # Ensure query_text is included in kwargs
        if 'query_text' not in kwargs:
            kwargs['query_text'] = query_text
            
        # Get raw search results
        raw_results = self.get_search_results(**kwargs)
        
        # Use the retriever's search method to format results
        return self.retriever.search(query_text=query_text, **kwargs).items
    
    @property
    def underlying_retriever(self):
        """Get the underlying retriever instance."""
        return self.retriever
