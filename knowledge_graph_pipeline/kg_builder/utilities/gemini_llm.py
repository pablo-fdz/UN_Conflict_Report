from neo4j_graphrag.llm import LLMInterface, LLMResponse
from langchain_google_genai import GoogleGenerativeAI
from typing import Any, Optional
import asyncio
from typing import Any, List, Optional, Union
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

class GeminiLLM(LLMInterface):
    """
    A custom LLM class for Google Gemini models that implements the Neo4j GraphRAG LLMInterface.
    """
    
    def __init__(
        self, 
        model_name: str, 
        google_api_key: str, 
        model_params: Optional[dict[str, Any]] = None,
        default_system_instruction: Optional[str] = None
    ):
        """
        Initialize the Gemini LLM.
        
        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-2.5-flash-preview-04-17")
            google_api_key: The Google API key to authenticate with Gemini
            model_params: Optional parameters to pass to the model (e.g., temperature)
            default_system_instruction: Default system prompt to use when none is provided
        """
        # Initialize the parent class
        super().__init__(model_name=model_name, model_params=model_params or {})
        
        # Store the API key
        self.google_api_key = google_api_key
        
        # Store the default system instruction
        self.default_system_instruction = default_system_instruction or "You are a helpful AI assistant."
        
        # Initialize the LangChain Gemini model
        self.llm = GoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.google_api_key,
            **self.model_params
        )
    
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invoke the Gemini model synchronously.
        
        Args:
            input: The text prompt to send to the model
            
        Returns:
            LLMResponse: An object containing the model's response
        """
        # Implement how to handle system_instruction
        effective_system_instruction = system_instruction or self.default_system_instruction
        
        try:
            # Get the response from the model
            response = self.llm.invoke(input)
            
            # Return as LLMResponse object (tokens_used is not provided by the Gemini API through LangChain)
            return LLMResponse(content=response)
        except Exception as e:
            # Handle any errors that might occur
            error_message = f"Error invoking Gemini model: {str(e)}"
            return LLMResponse(content=error_message)
    
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invoke the Gemini model asynchronously.
        
        Args:
            input: The text prompt to send to the model
            
        Returns:
            LLMResponse: An object containing the model's response
        """
        # Similar implementation for async version
        effective_system_instruction = system_instruction or self.default_system_instruction

        # Use run_in_executor to make the synchronous call asynchronous
        # This is because the LangChain GoogleGenerativeAI doesn't have native async support
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.invoke, input)
        return response