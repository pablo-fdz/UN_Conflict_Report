import sys
from pathlib import Path
from typing import Dict, List

# Utilities
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.llm import LLMInterface
from library.kg_builder.utilities import GeminiLLM
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG

# Neo4j and Neo4j GraphRAG imports
import neo4j

class AccuracyEvaluator:
    def __init__(self, base_claims_prompt: str, base_questions_prompt: str):

        # PROMPTS
        self.base_claims_prompt = base_claims_prompt
        self.base_questions_prompt = base_questions_prompt

    def _extract_verifiable_claims_one_section(self, llm: LLMInterface, section_text: str) -> str:
        """
        Extracts verifiable claims from the section text. Returns them as a list of strings.

        Args:
            llm: The language model to use for extraction.
            section_text (str): The text of the section from which to extract claims.
        """
        prompt = self.base_claims_prompt.format(section_text=section_text)
        claims_list = llm.invoke(prompt).content

        return claims_list

    def _generate_questions_one_section(self, llm: LLMInterface, claims_list: str) -> list:
        """
        Generates questions for each claim in the list. Returns them as a list of strings.

        Args:
            llm: The language model to use for generating questions.
            claims_list (list): A list of claims for which to generate questions.
        """

        prompt = self.base_questions_prompt.format(claims_list=claims_list)
        questions_dict = llm.invoke(prompt).content

        return questions_dict
    
    def get_claims_and_questions_one_section(self, section_text: str, llm_claims: LLMInterface, llm_questions: LLMInterface) -> str:
        """
        Evaluates a single section of text by extracting verifiable claims and generating questions.
        
        Args:
            section_text (str): The text of the section to evaluate.
            llm_claims (LLMInterface): The language model for extracting claims.
            llm_questions (LLMInterface): The language model for generating questions.
        """
        
        claims = self._extract_verifiable_claims_one_section(llm_claims, section_text)
        self.questions_dict = self._generate_questions_one_section(llm_questions, claims)

        return self.questions_dict
    
    def run_async(
            self, 
            query_text: str,
            llm_name: str,
            gemini_api_key: str,
            llm_params: Dict[str, str],
            retriever: Retriever,
            retriever_search_params: Dict[str, any] = None,
            rag_template: str = None,
            rag_system_instructions: str = None,
            examples: str = ''
        ):
        """
        Initializes the GraphRAG pipeline with the specified configurations.
        
        Args:
            query_text (str): The user question to search the knowledge graph.
            llm_name (str): The name of the LLM to use.
            gemini_api_key (str): The API key for the Gemini LLM.
            llm_params (Dict[str, str]): Parameters for the LLM.
            retriever (Retriever): The retriever to use for fetching relevant context.
            retriever_search_params (Dict[str, any], optional): Configuration for the search parameters of the input retriever. Defaults to None.
            rag_template (str, optional): Custom RAG template. Defaults to None (the default template of Neo4j).
            rag_system_instructions (str, optional): Custom system instructions for the RAG template. Defaults to None (default Neo4j system instructions).
            examples (str, optional): Examples to guide the LLM's response. Defaults to '' (empty string).
        """

        # Initialize LLM with GraphRAG configurations
        llm_graphrag = GeminiLLM(
            model_name=llm_name,
            google_api_key=gemini_api_key,
            model_params=llm_params
        )
        
        # Create RAGTemplate using configuration files
        rag_template = RagTemplate(
            template=rag_template,  # Use custom template if specified, otherwise use default
            expected_inputs=['query_text', 'context', 'examples'],  # Define expected inputs for the template
            system_instructions=rag_system_instructions,  # Use custom system instructions if specified, otherwise use default
        )
        
        graphrag = GraphRAG(
            llm=llm_graphrag,  # LLM for generating answers
            retriever=retriever,  # Retriever for fetching relevant context 
            prompt_template=rag_template  # RAG template for formatting the prompt
        )

        try:
            
            # Generate the answer using the GraphRAG pipeline
            graphrag_results = graphrag.search(
                query_text=query_text,  # User question that is used to search the knowledge graph (i.e., vector search and fulltext search is made based on this question); defaults to empty string if not provided
                message_history=None,  # Optional message history for conversational context (omitted for now)
                examples=examples,  # Optional examples to guide the LLM's response (defaults to empty string)
                retriever_config=retriever_search_params,  # Configuration for the search parameters of the input retriever
                return_context=False,  # Do not return the context used for generating the answer
            )
            
            # Get the generated answer from the GraphRAG results
            generated_answer = graphrag_results.answer

        except Exception as e:
            raise RuntimeError(f"Error during GraphRAG construction pipeline execution: {e}")
        
        return generated_answer  # Return the generated answer from the GraphRAG pipeline        