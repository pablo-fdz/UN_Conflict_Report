# Copyright (c) "Neo4j"
# Neo4j Sweden AB [https://neo4j.com]
# This code is largely based on the GraphRAG class from the neo4j_graphrag.generation module
# (version 1.7.0), with only minor modifications to separate the search query from the user query, and
# to enable the possibility to generate structured outputs. 

from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional, Union

from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    RagInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.generation.types import RagInitModel
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import LLMMessage, RetrieverResult
from neo4j_graphrag.utils.logging import prettify

from .types import RagSearchModel, RagResultModel  # Overwrite the original RagSearchModel to include search_text and query_text, and RagResultModel to consider different types of outputs (apart from strings).

logger = logging.getLogger(__name__)

class CustomGraphRAG:
    """Custom GraphRAG class that extends the functionality of the standard GraphRAG from
    neo4j_graphrag.generation, by separating the search query used for the retriever (i.e., 
    the query that will be embedded and/or based on which a full text search will be done) 
    from the user query (i.e., the question asked by the user and that will be used
    to generate the final result).
    
    Performs a GraphRAG search using a specific retriever and LLM.

    Example:

    .. code-block:: python

      import neo4j
      from neo4j_graphrag.retrievers import VectorRetriever
      from neo4j_graphrag.llm.openai_llm import OpenAILLM
      from neo4j_graphrag.generation import GraphRAG

      driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

      retriever = VectorRetriever(driver, "vector-index-name", custom_embedder)
      llm = OpenAILLM()
      graph_rag = GraphRAG(retriever, llm)
      graph_rag.search(query_text="Find me a book about Fremen")

    Args:
        retriever (Retriever): The retriever used to find relevant context to pass to the LLM.
        llm (LLMInterface): The LLM used to generate the answer.
        prompt_template (RagTemplate): The prompt template that will be formatted with context and user question and passed to the LLM.

    Raises:
        RagInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        # Initialization untouched from the original code
        try:
            validated_data = RagInitModel(
                retriever=retriever,
                llm=llm,
                prompt_template=prompt_template,
            )
        except ValidationError as e:
            raise RagInitializationError(e.errors())
        self.retriever = validated_data.retriever
        self.llm = validated_data.llm
        self.prompt_template = validated_data.prompt_template

    def search(
        self,
        search_text: str = "",  # New parameter to separate the search query from the user query
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: bool | None = True,
        structured_output: bool = False  # New parameter to enable structured outputs
    ) -> RagResultModel:
        # Method modified to separate search_text and query_text, and to consider 
        # the possibility of generating structured outputs.
        """

        This method performs a full RAG search:
            1. Retrieval: context retrieval with the `search_text`.
            2. Augmentation: prompt formatting with the retrieved context.
            3. Generation: answer generation with LLM with the `query_text`.

        Args:
            search_text (str): The text used to search for relevant context in the retriever.
                This text will be embedded and/or used for full text search.
            query_text (str): The user question used to generate the final answer.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            examples (str): Examples added to the LLM prompt.
            retriever_config (Optional[dict]): Parameters passed to the retriever.
                search method; e.g.: top_k
            return_context (bool): Whether to append the retriever result to the final result (default: False).
            structured_output (bool): Whether to enable structured outputs (default: False). 
                Only compatible with LLMs that support structured outputs and the response of which have a `parsed`
                attribute with the parsed data (see our GeminiLLM class in library.utilities.gemini_llm).

        Returns:
            RagResultModel: The LLM-generated answer.

        """
        if return_context is None:
            warnings.warn(
                "The default value of 'return_context' will change from 'False' to 'True' in a future version.",
                DeprecationWarning,
            )
            return_context = False
        try:
            validated_data = RagSearchModel(  # Modified to include new parameters
                search_text=search_text,
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config or {},
                return_context=return_context,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        query_text = self._build_query(validated_data.query_text, message_history)  # Modified simply by renaming the name of the created variable to `query_text`. Message history will be appended to the final query.
        retriever_result: RetrieverResult = self.retriever.search(
            query_text=search_text, **validated_data.retriever_config  # Modified to use the search_text as query_text
        )
        context = "\n".join(item.content for item in retriever_result.items)
        prompt = self.prompt_template.format(
            query_text=query_text, context=context, examples=validated_data.examples  # Now context is based on the search_text, while query_text is used for the final answer generation
        )
        logger.debug(f"RAG: retriever_result={prettify(retriever_result)}")
        logger.debug(f"RAG: prompt={prompt}")
        answer = self.llm.invoke(
            prompt,
            message_history,
            system_instruction=self.prompt_template.system_instructions,
        )
        # Structured output handling (newly added to support structured outputs)
        if structured_output == True:
            if not hasattr(answer, "parsed"):
                raise ValueError("The LLM response does not have a 'parsed' attribute. Ensure that the LLM supports structured outputs.")
            try:
                answer = answer.parsed  # Assuming the LLM response has a 'parsed' attribute
            except Exception as e:
                raise ValueError(f"Failed to parse the LLM response: {e}")
        else:
            if not hasattr(answer, "content"):
                raise ValueError("The LLM response does not have a 'content' attribute. Ensure that the LLM supports text generation.")
            if not isinstance(answer.content, str):
                raise ValueError("The LLM response content is not a string. Ensure that the LLM supports text generation.")
            try:
                answer = answer.content
            except Exception as e:
                raise ValueError(f"Failed to extract content from the LLM response: {e}")
        result: dict[str, Any] = {"answer": answer}  # Removed the .content attribute since the attribute that is extracted is handled above
        if return_context:
            result["retriever_result"] = retriever_result
        return RagResultModel(**result)

    def _build_query(
        self,
        query_text: str,
        message_history: Optional[List[LLMMessage]] = None,
    ) -> str:
        # Untouched method (since it is used to build the final query for invoking the LLM).
        # The _build_query method is designed exclusively to handle conversational history.
        # - If message_history exists: It summarizes the past conversation and prepends it to the current query_text. This creates a new, longer prompt for the LLM that includes the conversational context.
        # - If message_history is None: It does nothing and simply returns the original query_text unchanged.
        summary_system_message = "You are a summarization assistant. Summarize the given text in no more than 300 words."
        if message_history:
            summarization_prompt = self._chat_summary_prompt(
                message_history=message_history
            )
            summary = self.llm.invoke(
                input=summarization_prompt,
                system_instruction=summary_system_message,
            ).content
            return self.conversation_prompt(summary=summary, current_query=query_text)
        return query_text

    def _chat_summary_prompt(self, message_history: List[LLMMessage]) -> str:
        # Untouched method (since it is used to build the final query for invoking the LLM).
        message_list = [
            f"{message['role']}: {message['content']}" for message in message_history
        ]
        history = "\n".join(message_list)
        return f"""
Summarize the message history:

{history}
"""

    def conversation_prompt(self, summary: str, current_query: str) -> str:
        # Untouched method (since it is used to build the final query for invoking the LLM).
        return f"""
Message Summary:
{summary}

Current Query:
{current_query}
"""
