from typing import Optional, Generator, List
from dataclasses import dataclass

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from chains.retriever import RAGRetriever, get_retriever
from models.llm import OllamaLLM, get_ollama_llm
from prompts.rag_prompts import get_rag_prompt


@dataclass
class RAGResponse:
    answer: str
    sources: List[dict]
    query: str
    
    def __str__(self) -> str:
        return self.answer


class RAGChain:
    
    def __init__(self):
        self._retriever = get_retriever()
        self._llm = get_ollama_llm()
        self._prompt = get_rag_prompt()
        
        self._chain = self._build_chain()
    
    def _build_chain(self):
        """
        Build the LCEL chain for RAG.
        
        Returns:
            LCEL chain = question -> retrieve context -> format prompt -> generate
        """
        chain = (
            {
                "context": RunnableLambda(
                    lambda x: self._retriever.get_context_string(x["question"])
                ),
                "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            }
            | self._prompt
            | self._llm._llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
        filter: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Execute a RAG query and get a response.
        
        Args:
            question: User question
            filter: Optional metadata filter for retrieval
            
        Returns:
            RAGResponse with answer and sources
        """

        context = self._retriever.get_context_string(
            query=question,
            filter=filter,
            include_sources=True,
        )
        sources = self._retriever.get_sources(query=question, filter=filter)
        
        response = self._chain.invoke({"question": question})
        
        return RAGResponse(
            answer=response,
            sources=sources,
            query=question,
        )
    
    def stream_query(
        self,
        question: str,
        filter: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a RAG query response.
        
        Args:
            question: User question
            filter: Optional metadata filter
            
        Yields:
            Response chunks
        """
        context = self._retriever.get_context_string(
            query=question,
            filter=filter,
            include_sources=True,
        )
        
        
        prompt_value = self._prompt.format(
            context=context,
            question=question,
        )
        
        for chunk in self._llm.stream(prompt_value):
            yield chunk
    
    def query_simple(self, question: str) -> str:
        """
        Simple query returning just the answer string.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        response = self.query(question)
        return response.answer


def get_rag_chain() -> RAGChain:
    return RAGChain()
