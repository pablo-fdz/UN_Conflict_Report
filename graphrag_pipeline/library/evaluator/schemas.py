from pydantic import RootModel, BaseModel, Field
from typing import List, Dict
import enum

class Claims(BaseModel):
    """
    Represents a list of verifiable claims.
    The root of the model is a list of strings.
    """
    claims: List[str] = Field(
        description="A list of verifiable claims, where each claim is a self-contained, atomic statement that can be checked for accuracy."
    )

class QuestionsBase(BaseModel):
    """
    Represents a dictionary of claims (keys) and questions (values).
    The root of the model is a dictionary where each key is a claim and the value 
    is a list of questions related to that claim.
    """
    claim: str = Field(
        description="A verifiable claim for which questions are being asked."
        )
    questions: List[str] = Field(
        description="A list of questions related to the claim."
        )

class Questions(BaseModel):
        """
        A dictionary cannot be directly used as input to the Gemini API, so this 
        class is used to wrap the dictionary in a Pydantic model. See these 
        links for incompatibility issues with dicts:
        - https://github.com/googleapis/python-genai/issues/460;
        - https://github.com/googleapis/python-genai/issues/70).
        """
        c_and_a_list: list[QuestionsBase]

class GraphRAGResultsBase(BaseModel):
    question: str = Field(
        description="A question that was asked about the claim."
    )
    answer: str = Field(
        description="The answer provided in response to the question."
    )
    source: str = Field(
        description="The source of the answer, which must be in the format `<author (physical person or newspaper>: <URL or reference to a document>`."
    )

class GraphRAGResults(BaseModel):
    """
    Represents the results of a GraphRAG evaluation.
    This model contains a list of questions and their corresponding answers.
    """
    results: List[GraphRAGResultsBase] = Field(
        description="A list of questions, answers and sources related to the claims."
    )

class EvaluationConclusions(enum.Enum):
    """Enumeration for evaluation conclusions."""
    TRUE = "true"
    FALSE = "false"
    MIXTURE = "mixture"

class EvaluationResults(BaseModel):
    """
    Represents the results of an evaluation.
    This model contains the conclusion of the evaluation and a justification for that conclusion.
    """
    conclusion: EvaluationConclusions = Field(
        description="The conclusion of the evaluation, indicating whether the claims are true, false, or a mixture of both. This field is REQUIRED."
        )
    justification: str = Field(
        description="A detailed explanation of the evaluation conclusion, including the reasoning behind it. This field is REQUIRED if the conclusion is 'false' or 'mixture'. If the conclusion is 'true', this field can be an empty string."
        )