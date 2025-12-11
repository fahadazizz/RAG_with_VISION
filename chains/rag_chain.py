from typing import Optional, Generator, List
from dataclasses import dataclass

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from chains.retriever import get_retriever
from models.llm import get_ollama_llm
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
            LCEL chain = prompted LLM
        """
        # We pass "context" and "question" directly to invoke()
        chain = (
            self._prompt
            | self._llm._llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Execute a RAG query and get a response.
        
        Args:
            question: User question
            
        Returns:
            RAGResponse with answer and sources
        """
        # 1. Retrieve documents (Single Pass)
        results = self._retriever.retrieve(query=question)
        
        # 2. Format context
        context = self._retriever.format_context(results)
        
        # 3. Extract sources
        sources = [result.metadata for result in results]
        
        # 4. Generate answer
        response = self._chain.invoke({
            "context": context,
            "question": question
        })
        
        return RAGResponse(
            answer=response,
            sources=sources,
            query=question,
        )
    
    def stream_query(
        self,
        question: str,
    ) -> Generator[str, None, None]:
        """
        Stream a RAG query response.
        
        Args:
            question: User question
            
        Yields:
            Response chunks
        """
        # 1. Retrieve & Format
        results = self._retriever.retrieve(query=question)
        context = self._retriever.format_context(results)
        
        # 2. stream
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
