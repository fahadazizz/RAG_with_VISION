from typing import Optional
from functools import lru_cache

from langchain_ollama import ChatOllama

from config import get_settings


class OllamaLLM:
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ):
        settings = get_settings()
        
        self.model_name = model_name or settings.llm_model_name
        self.base_url = base_url or settings.llm_base_url
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.num_ctx = num_ctx or 4096
        
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )
    
    def invoke(self, prompt: str) -> str:
        """
        Generate a response for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        response = self._llm.invoke(prompt)
        return response.content
    
    def stream(self, prompt: str):
        """
        Stream a response for a prompt.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Response chunks
        """
        for chunk in self._llm.stream(prompt):
            yield chunk.content

def get_ollama_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> OllamaLLM:
    return OllamaLLM(
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
    )


@lru_cache()
def get_default_llm() -> OllamaLLM:
    """
    Get a cached default LLM instance.
    
    Returns:
        Cached OllamaLLM instance
    """
    return OllamaLLM()
