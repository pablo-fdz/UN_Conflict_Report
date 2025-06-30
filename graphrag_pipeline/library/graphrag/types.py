from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, field_validator

from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult

class RagSearchModel(BaseModel):
    search_text: str
    query_text: str
    examples: str = ""
    retriever_config: dict[str, Any] = {}
    return_context: bool = True

class RagResultModel(BaseModel):
    answer: Union[str, Any]  # Can be a string or any other type, e.g., structured output
    retriever_result: Union[RetrieverResult, None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
