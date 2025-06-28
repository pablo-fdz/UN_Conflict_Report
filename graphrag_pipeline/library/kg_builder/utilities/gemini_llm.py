"""
Useful documentation:
- google-genai library: https://googleapis.github.io/python-genai/index.html
- API documentation of neo4j-graphrag for design of custom LLMInterface: https://neo4j.com/docs/neo4j-graphrag-python/current/api.html#llminterface
- User guide of neo4j-graphrag for design of custom LLMInterface: https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html#using-a-custom-model
- Avoiding resource exhaustion with the Google Gemini API and tenacity: https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-handle-429-resource-exhaustion-errors-in-your-llms.
    - More solutions here: https://support.google.com/gemini/thread/343007251/resource-exhausted-though-to-my-knowledge-staying-within-quotas?hl=en
"""

from typing import Any, List, Optional, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import the official Google Generative AI library
from google import genai
from google.api_core.exceptions import ResourceExhausted

# Import necessary components from the neo4j-graphrag library
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

class GeminiLLMResponse(LLMResponse):
    """
    A custom response object that extends the base LLMResponse to include
    the parsed Pydantic model from Gemini's structured output (apart from the `.content` field, 
    which must ALWAYS be a string - see LLMResponse class). Adding this field should 
    not break compatibility with existing code that uses LLMResponse,
    as the functions and classes from neo4j-graphrag expect an LLMResponse object
    with a `.content` field (a str) containing the text response from the model (which is 
    extracted from the Gemini model's response). See for example the usage in the
    GraphRAG class in the neo4j-graphrag library.
    """
    parsed: Optional[Any] = None

class GeminiLLM(LLMInterface):
    """
    A custom LLM class for Google Gemini models that implements the Neo4j GraphRAG LLMInterface.

    This implementation uses the `google-genai` library, which provides a unified client
    for both Gemini and Vertex AI APIs. It is designed to be a drop-in replacement for
    other LLM interfaces within the neo4j-graphrag framework.

    Example usage:
    .. code-block:: python
        from graphrag_pipeline.library.kg_builder.utilities.gemini_llm import GeminiLLM
        from pydantic import RootModel, Field
        from typing import List

        # Create a pydantic model for structured output.
        class Claims(RootModel[List[str]]):
            root: List[str] = Field(
                description="A list of verifiable claims, where each claim is a self-contained, atomic statement that can be checked for accuracy."
            )

        # Initialize the Gemini LLM client with your model name and Google API key.
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            google_api_key="YOUR_GOOGLE_API_KEY",
            model_params={
                "temperature": 0.7, 
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "response_schema": Claims
            },
            default_system_instruction="You are a helpful assistant."
        )

        # Invoke the model with a user prompt.
        response = llm.invoke(input="What are the verifiable claims in the following text? 'The capital of France is Paris. The Eiffel Tower is in Paris.'")

        # The response will contain the text and parsed data.
        print(response.content)  # The text response from the model.
        print(response.parsed)   # The parsed structured output, if applicable. If the model's response is structured, this will contain the parsed data, in the type of the Claims model.
    """

    def __init__(
        self,
        model_name: str,
        google_api_key: str,
        model_params: Optional[dict[str, Any]] = None,
        default_system_instruction: Optional[str] = None,
    ):
        """
        Initializes the Gemini LLM client.

        Args:
            model_name (str): The name of the Gemini model to use (e.g., "gemini-1.5-flash-latest").
            google_api_key (str): The Google API key for authentication.
            model_params (Optional[dict[str, Any]]): A dictionary of optional parameters to pass
                to the model during generation. This can include `temperature`, `max_output_tokens`,
                `response_mime_type`, etc. Defaults to None.
            default_system_instruction (Optional[str]): A default system-level instruction to
                guide the model's behavior. This can be overridden on a per-call basis.
                Defaults to None.
        """
        # Initialize the parent class with model_name and model_params
        super().__init__(model_name=model_name, model_params=model_params or {})

        # Initialize the synchronous client using the provided API key.
        # The async client is accessed via the .aio property of this client instance.
        self.client = genai.Client(api_key=google_api_key)

        # Store the default system instruction for reuse.
        self.default_system_instruction = default_system_instruction

    def _get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> List:
        """
        Constructs and formats the conversation history for the Google GenAI API.

        The Google API expects a list of dictionaries, where each dictionary represents a
        message with a 'role' ('user' or 'model') and 'parts'. This method converts the
        `LLMMessage` or `MessageHistory` objects into this required format.

        Args:
            input (str): The latest user prompt to be added to the conversation.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): The existing
                conversation history.

        Returns:
            List: A list of messages formatted for the API.
        """
        messages = []
        if message_history:
            # Determine if the history is a MessageHistory object or a simple list
            history = (
                message_history.messages
                if isinstance(message_history, MessageHistory)
                else message_history
            )
            for msg in history:
                # The Google API uses 'model' for the assistant's role, while GraphRAG uses 'assistant'.
                # We map 'assistant' to 'model' and keep 'user' as is.
                role = "model" if msg.role == "assistant" else msg.role
                # Ensure we only process roles the API understands ('user' and 'model').
                if role in ["user", "model"]:
                    messages.append({"role": role, "parts": [{"text": msg.content}]})

        # The final user input is always appended as the last message in the sequence.
        messages.append({"role": "user", "parts": [{"text": input}]})
        return messages

    # Retry "Randomly wait up to 2^x * 1 seconds between each retry until the
    # range reaches 60 seconds, then randomly up to 60 seconds afterwards"
    # We will retry up to 5 times on ResourceExhausted errors.
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invokes the Gemini model synchronously with a given prompt and conversation history.
        Includes a retry mechanism for handling resource exhaustion errors.

        Args:
            input (str): The text prompt to send to the model.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): The past
                conversation history.
            system_instruction (Optional[str]): An optional system instruction to override the
                default for this specific call.

        Returns:
            LLMResponse: An object containing the model's response text.

        Raises:
            LLMGenerationError: If an error occurs during the API call.
        """
        # Prepare the generation configuration.
        config_params = self.model_params.copy()
        current_system_instruction = system_instruction or self.default_system_instruction
        if current_system_instruction:
            # The system_instruction is part of the GenerateContentConfig
            config_params["system_instruction"] = current_system_instruction

        generation_config = genai.types.GenerateContentConfig(**config_params)
        messages = self._get_messages(input, message_history)

        try:
            # Make the synchronous API call using the client.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=generation_config,
            )
            # Extract the text and parsed data from the response.
            parsed_data = getattr(response, 'parsed', None)  # Check if the response has parsed data, defaults to None
            return GeminiLLMResponse(content=response.text, parsed=parsed_data)
        except ResourceExhausted as e:
            # Re-raise the specific exception to be caught by tenacity for retry
            raise e
        except Exception as e:
            # If any other exception occurs, wrap it in LLMGenerationError as expected by the interface.
            raise LLMGenerationError(f"Error invoking Gemini model: {e}") from e

    # Retry "Randomly wait up to 2^x * 1 seconds between each retry until the
    # range reaches 60 seconds, then randomly up to 60 seconds afterwards"
    # We will retry up to 5 times on ResourceExhausted errors.
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invokes the Gemini model asynchronously with a given prompt and conversation history.
        Includes a retry mechanism for handling resource exhaustion errors.

        This method uses the native async client for efficient, non-blocking API calls.

        Args:
            input (str): The text prompt to send to the model.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): The past
                conversation history.
            system_instruction (Optional[str]): An optional system instruction to override the
                default for this specific call.

        Returns:
            LLMResponse: An object containing the model's response text.

        Raises:
            LLMGenerationError: If an error occurs during the API call.
        """
        # Prepare the generation configuration.
        config_params = self.model_params.copy()
        current_system_instruction = system_instruction or self.default_system_instruction
        if current_system_instruction:
            # The system_instruction is part of the GenerateContentConfig
            config_params["system_instruction"] = current_system_instruction

        generation_config = genai.types.GenerateContentConfig(**config_params)
        messages = self._get_messages(input, message_history)

        try:
            # Make the asynchronous API call using the client's aio property.
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=generation_config,
            )
            # Extract the text and parsed data from the response.
            parsed_data = getattr(response, 'parsed', None)
            return GeminiLLMResponse(content=response.text, parsed=parsed_data)
        except ResourceExhausted as e:
            # Re-raise the specific exception to be caught by tenacity for retry
            raise e
        except Exception as e:
            # If any exception occurs, wrap it in LLMGenerationError.
            raise LLMGenerationError(f"Error invoking Gemini model asynchronously: {e}") from e